import os
import sys

# Windows mathematical library compatibility fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Explicitly import numpy 1.x before torch to ensure correct DLL initialization on Windows
import numpy 
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import DRModel 
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
MODEL = None
MODEL_PATH = "best_ordinal_model.pt"

# Normalization same as training
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
        ]

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_tensor, target_stage):
        self.model.train() # Set to train to enable gradient tracking if needed, though we only need grads of input
        logits = self.model(input_tensor)
        self.model.zero_grad()
        
        # Target the logit corresponding to the predicted stage or adjacent
        target = logits[:, min(target_stage, 3)]
        target.backward(retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        self.model.eval() # Set back to eval
        return cam

def load_model():
    global MODEL
    if MODEL is not None:
        return

    path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    if not os.path.exists(path):
        print(f"Model file not found at {path}")
        return

    try:
        print(f"Loading model from {path}...")
        # Instantiate the model architecture
        model_instance = DRModel()
        
        # Load weights
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Handle if checkpoint is state_dict (which we expect) or full model
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            # strict=False allows loading if there are minor mismatches (like missing head keys or extra keys)
            # helping us debug if architecture isn't perfect
            missing, unexpected = model_instance.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys: {missing[:5]}...")
            if unexpected:
                print(f"Warning: Unexpected keys: {unexpected[:5]}...")
                
        else:
            # Fallback if it was a full model
            model_instance = checkpoint

        model_instance.eval()
        MODEL = model_instance
        print("Success: Model loaded.")

    except Exception as e:
        print(f"Failed to load model: {e}")
        MODEL = None

def predict_image(image_bytes):
    """
    Predicts DR severity using the loaded PyTorch model.
    """
    load_model()
    
    if MODEL is None:
        return {
            "severity": 0,
            "confidence": 0.0,
            "label": "Model Error (See Logs)"
        }

    try:
        # 1. Prepare Image
        image_stream = io.BytesIO(image_bytes)
        original_image = Image.open(image_stream).convert('RGB')
        img_tensor = inference_transform(original_image).unsqueeze(0) 

        # 2. Basic Prediction
        with torch.no_grad():
            logits = MODEL(img_tensor)
            probs = torch.sigmoid(logits)
            pred_tensor = (probs > 0.5).sum(dim=1)
            severity = int(pred_tensor[0].item())
            
            p = probs.squeeze().tolist()
            if severity == 0:
                confidence = 1.0 - p[0]
            elif severity == 4:
                confidence = p[3]
            else:
                confidence = p[severity-1] * (1.0 - p[severity])

        # 3. Generate Grad-CAM Heatmap
        heatmap_base64 = None
        if severity > 0: 
            try:
                # Target the 1x1 projection layer - it's the last spatial layer before Transformer
                target_layer = MODEL.proj
                cam_generator = GradCAM(MODEL, target_layer)
                
                # We need to enable gradients for CAM
                img_tensor.requires_grad = True
                heatmap = cam_generator.generate(img_tensor, severity)
                
                # Process heatmap with OpenCV
                heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Overlay on original image
                original_np = np.array(original_image)
                overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)
                
                # Convert back to PIL & Base64
                overlay_image = Image.fromarray(overlay)
                buffered = io.BytesIO()
                overlay_image.save(buffered, format="JPEG", quality=85)
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            except Exception as cam_e:
                print(f"CAM generation error: {cam_e}")

        return {
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "label": get_severity_label(severity),
            "heatmap": heatmap_base64
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            "severity": 0,
            "confidence": 0.0,
            "label": "Error",
            "heatmap": None
        }

def get_severity_label(severity):
    labels = {
        0: "No_DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferate_DR",
    }
    return labels.get(severity, "Unknown")

import os
from groq import Groq

# Initialize Groq client dynamically
# Ensure GROQ_API_KEY is set in environment variables

def get_chat_response(message: str, prediction_result: dict = None):
    """
    Generates a response using the Groq API (Llama 3).
    Includes strict system prompting for medical safety and optional patient context.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return "I apologize, but I am not configured correctly. Please contact the administrator."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build context prompt based on available data
        context_prompt = ""
        if isinstance(prediction_result, dict):
            severity = prediction_result.get("label", "Unknown")
            confidence = prediction_result.get("confidence", 0)
            context_prompt = f"The patient just received a screening result: {severity} with {confidence*100}% confidence. "
        elif isinstance(prediction_result, str):
            context_prompt = f"The patient is reviewing an AI-generated analysis of their medical report: {prediction_result[:1000]} "

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful and professional medical assistant for a Diabetic Retinopathy screening platform.
                    
                    {context_prompt}
                    
                    RULES:
                    1. Provide educational information about Diabetic Retinopathy (symptoms, stages, prevention, treatments) based on general medical knowledge.
                    2. If a result is provided above, you can explain what it means in general terms but keep emphasis on seeing a doctor.
                    3. DO NOT provide personal medical advice, diagnoses, or treatment plans for specific individuals.
                    4. If a user asks for a diagnosis based on symptoms, strictly advise them to consult an eye care professional (ophthalmologist).
                    5. Keep answers concise, empathetic, and easy to understand.
                    6. If asked about the screening tool, explain it uses AI to detect signs of DR but is not a replacement for a doctor.
                    
                    Tone: Professional, calm, reassuring, and objective.
                    """
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.5,
            max_tokens=300,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "I apologize, but I'm having trouble connecting to my knowledge base right now. Please try again later."

def search_eye_hospitals(city: str = None, lat: float = None, lng: float = None):
    """
    Uses SerpApi to find live information about specialized eye hospitals.
    Can search by city name or geographic coordinates.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key or "YOUR_" in api_key:
        print("SerpApi Key missing or placeholder.")
        return []

    try:
        from serpapi import GoogleSearch
        
        if lat is not None and lng is not None:
            # Use Google Maps engine for better proximity results when coordinates are provided
            params = {
                "engine": "google_maps",
                "q": "eye hospitals retina centers",
                "ll": f"@{lat},{lng},12z",
                "hl": "en",
                "type": "search",
                "api_key": api_key
            }
        else:
            params = {
                "q": f"top eye hospitals retina centers in {city} India",
                "location": f"{city}, India",
                "hl": "en",
                "gl": "in",
                "google_domain": "google.co.in",
                "api_key": api_key
            }

        def execute_search(search_params):
            try:
                search = GoogleSearch(search_params)
                res = search.get_dict()
                if isinstance(res, dict) and "error" in res:
                    return res, False
                return res, True
            except Exception as se:
                print(f"SerpApi Low Level Error: {se}")
                return {"error": str(se)}, False

        results, success = execute_search(params)
        
        # Fallback if location is unsupported or other error
        if not success:
            error_msg = results.get("error", "").lower()
            if "location" in error_msg or "unsupported" in error_msg or "expecting value" in error_msg:
                print(f"Attempting fallback search due to error: {error_msg}")
                # Remove specific location/ll and try a broad query
                fallback_params = {
                    "q": params["q"],
                    "hl": "en",
                    "gl": "in",
                    "api_key": api_key
                }
                if city:
                    fallback_params["q"] = f"top eye hospitals retina centers in {city} India"
                elif lat and lng:
                    fallback_params["q"] = "top eye hospitals retina centers near me"
                
                results, success = execute_search(fallback_params)

        if not success:
            print(f"SerpApi Search failed after fallback: {results.get('error')}")
            return []

        hospitals = []
        
        # 1. Try Local Results (Standard Local Pack or Maps Engine)
        local = results.get("local_results", [])
        if isinstance(local, dict):
            # Sometimes nested under 'places' or similar
            local = local.get("places", []) or list(local.values())[0] if local.values() else []
        
        if isinstance(local, list):
            for place in local:
                if not isinstance(place, dict): continue
                hospitals.append({
                    "name": str(place.get("title") or place.get("name") or ""),
                    "address": str(place.get("address") or place.get("snippet") or city or ""),
                    "type": str(place.get("type") or place.get("category") or "Eye Hospital"),
                    "rating": place.get("rating"),
                    "link": str(place.get("links", {}).get("website") or place.get("website") or place.get("link") or "")
                })

        # 2. Try Place Results (If it's a single point of interest)
        if not hospitals and "place_results" in results:
            place = results["place_results"]
            if isinstance(place, dict):
                hospitals.append({
                    "name": str(place.get("title") or ""),
                    "address": str(place.get("address") or ""),
                    "type": str(place.get("type") or "Eye Hospital"),
                    "rating": place.get("rating"),
                    "link": str(place.get("website") or "")
                })

        # 3. Try Organic Results but only for VERY specific entities if local is missing
        # We look for organic results that look like they might be specific hospital sites
        if not hospitals and "organic_results" in results:
            organic = results["organic_results"]
            if isinstance(organic, list):
                for res in organic[:5]:
                    if not isinstance(res, dict): continue
                    title = res.get("title", "")
                    # Filter out "List of..." or "Top 10..." aggregator sites if possible
                    if any(x in title.lower() for x in ["list of", "top 10", "best 5", "directory"]):
                        continue

                    hospitals.append({
                        "name": str(title),
                        "address": str(res.get("snippet") or city or ""),
                        "type": "Specialized Center",
                        "link": str(res.get("link") or "")
                    })

        # Final sanitization
        clean_hospitals = []
        for h in hospitals:
            if h.get("name") and len(h["name"]) > 3:
                # Avoid duplicate names
                if not any(ch["name"] == h["name"] for ch in clean_hospitals):
                    clean_hospitals.append(h)

        return clean_hospitals
        
    except Exception as e:
        print(f"SerpApi Search Error: {e}")
        # Log the keys of results to help debug if it fails again
        if 'results' in locals():
            print(f"Results keys: {list(results.keys())}")
        return []

def extract_text_from_pdf(pdf_bytes):
    """
    Extracts text from a PDF file using pypdf.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        return ""

def get_pdf_analysis(pdf_text: str):
    """
    Analyzes medical report text using Groq to provide a patient-friendly summary.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Analysis configuration missing."

    if not pdf_text or len(pdf_text) < 10:
        return "Could not extract sufficient text from the report. Please ensure the PDF contains readable text."

    try:
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a compassionate medical report assistant. 
                    Your goal is to help a patient understand their medical report (specifically related to eye care or general health).
                    
                    RULES:
                    1. Provide a "Complete Info Review" of the extracted text.
                    2. Use Markdown TABLES to compare findings or organize data where possible (e.g., Section | Finding | Meaning).
                    3. Use EMOJIS/ICONS to make sections more readable (e.g., 🩺, 📖, ⚠️).
                    4. Explain medical terms in simple, easy-to-understand language.
                    5. Highlight key findings, status of Diabetic Retinopathy (if mentioned), and any "Action Items" suggested by the report.
                    6. ALWAYS include a prominent disclaimer that this is an AI interpretation and they MUST discuss it with their doctor.
                    7. If the report mentions specific grades or stages, explain them based on standard medical definitions.
                    8. Tone: Reassuring, clear, and professional.
                    """
                },
                {
                    "role": "user",
                    "content": f"Review this medical report text and provide a patient-friendly summary:\n\n{pdf_text[:4000]}" # Limit to 4k chars for prompt safety
                }
            ],
            temperature=0.3, # Lower temperature for more factual summary
            max_tokens=600,
            stream=False,
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Groq PDF Analysis Error: {e}")
        return "I'm sorry, I encountered an error while analyzing your report. Please try again or consult your doctor."
