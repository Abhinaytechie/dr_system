import os
from fpdf import FPDF
from datetime import datetime
import base64
import io
from PIL import Image

class DiagnosticReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(20, 184, 166) # Medical teal
        self.cell(self.epw, 15, 'DR-System Screening Report', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100)
        self.cell(self.epw, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

    def footer(self):
        self.set_y(-25)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150)
        self.cell(self.epw, 10, 'DISCLAIMER: This is an AI-generated screening result for educational purposes only.', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(self.epw, 5, 'It does not replace a clinical diagnosis by an ophthalmologist.', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_y(-15)
        self.cell(self.epw, 10, f'Page {self.page_no()}', border=0, align='C')

def generate_pdf_report(prediction_result: dict, user_message: str = ""):
    """
    Generates a PDF report for the patient.
    """
    pdf = DiagnosticReport(orientation='P', unit='mm', format='A4')
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    # 1. Summary Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30)
    pdf.cell(pdf.epw, 10, 'Screening Summary', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Details table - Proportional centering
    # Total width 180 (Full EPW)
    w1, w2 = 40, 50 
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(245, 245, 245)
    
    label = str(prediction_result.get("label", "Unknown"))
    conf = f'{prediction_result.get("confidence", 0)*100:.1f}%'

    # Shift table to start at margin explicitly
    pdf.set_x(15)
    pdf.cell(w1, 10, ' Severity Stage:', border=1, align='L', fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(w2, 10, f' {label}', border=1, align='L')
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(w1, 10, ' AI Confidence:', border=1, align='L', fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(w2, 10, f' {conf}', border=1, align='L', new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)

    # 2. Visual Insight (Heatmap)
    if prediction_result.get("heatmap"):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(30)
        pdf.cell(pdf.epw, 10, 'Diagnostic Heatmap (Grad-CAM)', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100)
        pdf.multi_cell(pdf.epw, 5, 'The heatmap highlights regions in the retinal image that influenced the AI detection. Intensive colors (yellow/red) indicate areas where DR associated markers were localized.', align='C')
        pdf.ln(8)
        
        try:
            heatmap_data = base64.b64decode(prediction_result["heatmap"])
            heatmap_img = Image.open(io.BytesIO(heatmap_data))
            temp_path = "temp_heatmap_report.jpg"
            heatmap_img.save(temp_path)
            
            img_w = 125 # Larger, centered image
            pdf.image(temp_path, x=(210-img_w)/2, w=img_w)
            pdf.ln(8)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            pdf.set_text_color(200, 0, 0)
            pdf.cell(pdf.epw, 10, f'Notice: Image preview unavailable ({str(e)})', border=0, align='C', new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    
    # 3. Recommendations
    pdf.set_text_color(30)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(pdf.epw, 10, 'Clinical Recommendations', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 10)
    
    recommendations = [
        "Consult an ophthalmologist for a comprehensive clinical exam.",
        "Maintain strict control of blood sugar, blood pressure, and cholesterol levels.",
        "Ensure regular annual screening for both eyes.",
        "If you experience sudden blurred vision or dark spots, seek emergency care."
    ]
    
    pdf.set_draw_color(220)
    for rec in recommendations:
        pdf.set_x((210 - 160) / 2) # Center the recommendation blocks
        pdf.multi_cell(160, 10, f"- {str(rec)}", border='B', align='L')
    
    # Explicitly return bytes to avoid FastAPI encoding issues with bytearray
    return bytes(pdf.output())
