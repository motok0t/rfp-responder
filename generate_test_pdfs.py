from fpdf import FPDF

# Create the "old" RFP (with answers)
pdf_old = FPDF()
pdf_old.add_page()
pdf_old.set_font("Arial", size=12)
old_text = """1. What is your company's primary focus?
We specialize in building secure web applications and cloud infrastructure for healthcare clients.

2. How do you ensure data security?
We use AES-256 encryption at rest and TLS 1.3 in transit. All employees undergo annual security training.

3. How many years of experience do you have?
Our team has over 10 years of combined experience in enterprise software development."""
pdf_old.multi_cell(0, 10, old_text)
pdf_old.output("old_rfp.pdf")

# Create the "new" RFP (questions only)
pdf_new = FPDF()
pdf_new.add_page()
pdf_new.set_font("Arial", size=12)
new_text = """1. What is your company's main area of expertise?
2. What security protocols do you use to protect client data?
3. What is the total experience of your development team?"""
pdf_new.multi_cell(0, 10, new_text)
pdf_new.output("new_rfp.pdf")

print("PDF files created successfully!")
