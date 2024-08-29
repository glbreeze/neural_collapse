import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

# Load the PDF file
pdf_document = "nc1nc2_stl10.pdf"
pdf = fitz.open(pdf_document)

# Extract the first page as an example (0-indexed)
page = pdf.load_page(0)
pix = page.get_pixmap()

# Convert the extracted page to a byte array
img_data = pix.tobytes("png")

# Load the image data into Matplotlib
img = plt.imread(io.BytesIO(img_data))

# Create a figure and axis
fig, ax = plt.subplots()

# Display the image
ax.imshow(img)
ax.axis('off')  # Hide the axes

# Modify the font size (example)
plt.rcParams.update({'font.size': 14})

# Save the modified plot to a new PDF
output_pdf = "modified_plot.pdf"
pdf_pages = PdfPages(output_pdf)
pdf_pages.savefig(fig)
pdf_pages.close()

plt.show()