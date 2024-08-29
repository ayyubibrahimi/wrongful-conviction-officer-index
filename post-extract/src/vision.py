import os
import json
from io import BytesIO
import base64
import logging
import numpy as np
import cv2
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage
import csv
import hashlib

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TARGET_IMAGE_SIZE = (800, 800)
MAX_FILE_SIZE = 2 * 1024 * 1024

INPUT_DIR = "../data/input"
OUTPUT_DIR = "../data/output"

def generate_hash_id(officer_name, officer_sex, agency_name, status, hire_date, separation_date, reason):
    """Generate a unique hash ID for each entry."""
    # Combine all fields, using lowercase and removing extra spaces
    combined_info = "|".join([
        officer_name.lower().strip(),
        officer_sex.lower().strip(),
        agency_name.lower().strip(),
        status.lower().strip(),
        hire_date.strip(),
        separation_date.strip(),
        reason.lower().strip()
    ])
    # Create a hash of the combined information
    return hashlib.md5(combined_info.encode()).hexdigest()

def generate_person_id(officer_name, officer_sex):
    """Generate a consistent ID for linking entries of the same person."""
    combined_info = "|".join([
        officer_name.lower().strip(),
        officer_sex.lower().strip()
    ])
    return hashlib.md5(combined_info.encode()).hexdigest()

def parse_description(description, filename, page_number):
    lines = description.split("\n")
    parsed_data = []
    current_entry = {}
    officer_info = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Officer Name:"):
            officer_info["Officer Name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Officer Sex:"):
            officer_info["Officer Sex"] = line.split(":", 1)[1].strip()
        elif line.startswith("Agency Name:"):
            if current_entry:
                parsed_data.append({**officer_info, **current_entry, "Filename": filename, "Page Number": page_number})
            current_entry = {"Agency Name": line.split(":", 1)[1].strip()}
        elif ":" in line:
            key, value = line.split(":", 1)
            current_entry[key.strip()] = value.strip() if value.strip() else "N/A"

    if current_entry:
        parsed_data.append({**officer_info, **current_entry, "Filename": filename, "Page Number": page_number})

    return parsed_data


def process_pdf(pdf_path):
    """Process the top 1/3 of each page in a PDF file and get a description of each page"""
    output_data = {"messages": []}
    filename = os.path.basename(pdf_path)

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)
        for page_number in range(page_count):
            try:
                images = convert_from_path(
                    pdf_path, first_page=page_number + 1, last_page=page_number + 1
                )
                image = images[0]

                width, height = image.size
                cropped_image = image.crop((0, 0, width, height // 3))

                resized_image = resize_image(cropped_image)

                preprocessed_image = preprocess_image(resized_image)

                base64_image = encode_image_with_size_control(
                    preprocessed_image, MAX_FILE_SIZE
                )

                description = get_page_description(base64_image)

                output_data["messages"].append(
                    {"page_number": page_number + 1, "description": description, "filename": filename}
                )
                logging.info(f"Processed top 1/3 of page {page_number+1} of {pdf_path}")
            except Exception as e:
                logging.warning(
                    f"Error processing top 1/3 of page {page_number+1} of {pdf_path}: {str(e)}"
                )
                output_data["messages"].append(
                    {
                        "page_number": page_number + 1,
                        "description": f"Error processing page: {str(e)}",
                        "filename": filename
                    }
                )

    return output_data



def preprocess_image(image):
    """Apply advanced preprocessing techniques to enhance overall image quality, focusing on black text on white background"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Convert to RGB if image is in RGBA mode
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # Convert to LAB color space for more accurate color processing
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel to improve overall contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding to separate text from background
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Apply bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)

    # Adjust gamma to enhance details in darker regions
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    gamma_corrected = cv2.LUT(smooth, table)

    # Convert back to PIL Image
    processed_image = Image.fromarray(gamma_corrected)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.5)

    # Enhance sharpness one more time
    sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
    final_image = sharpness_enhancer.enhance(1.5)

    return final_image


def resize_image(image):
    """Resize the image to fit within the target size while maintaining aspect ratio"""
    image.thumbnail(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image


def encode_image_with_size_control(image, max_size, initial_quality=85):
    """Encode an image to base64 with size control"""
    quality = initial_quality
    while quality > 20:  # Set a lower bound for quality
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        if len(img_str) <= max_size:
            return img_str.decode("utf-8")
        quality -= 5
    raise ValueError(
        "Unable to compress image to required size while maintaining acceptable quality"
    )


def get_page_description(base64_image):
    """Get a description of the page using Claude Haiku"""
    chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)

    prompt = """
    <task_description>
    As a Data Extraction Specialist, your task is to accurately extract and format the employment history information from the provided document. Focus solely on the employment history table at the top of the document, ignoring all other tables and information.
    </task_description>
    
    <guidelines>
    1. Extract all essential information from the employment history table only.
    2. Present the information in a structured format as specified below.
    3. Ensure accuracy and completeness of the extracted data.
    4. DO NOT include any details not explicitly stated in the employment history table.
    5. Use the provided format to organize the extracted information.
    </guidelines>
    
    <essential_information>
    For each row in the employment history table, extract the following information: 
    a. Officer Name
    b. Officer Sex 
    c. Agency Name
    d. Employment Status
    e. Hire Date
    f. Separation Date
    g. Reason for Separation

    Repeat the above five fields for each employment entry in the table, including the any duplicate entries.
    Some, of the fields may be blank. If they are blank, fill the field as "N/A".

    </essential_information>
    
    <output_format>
    For each row in the employment history table, extract the following information: 
    Officer Name:
    Officer Sex: (if available)
    
    For each employment entry:

    Agency Name:
    Status:
    Hire Date:
    Separation Date: 
    Reason for Separation: 

    Repeat the above five fields for each employment entry in the table, including the any duplicate entries. 
    </output_format>

    <warnings>
    - Do not extract information from any table other than the employment history table.
    - Avoid including speculative information or drawing conclusions not explicitly stated in the table.
    - If a field is empty in the original table, leave it blank in your response.
    - For current employments, leave the Separation Date and Reason for Separation blank.
    - If there's only one employment entry, do not number the fields.
    </warnings>

    <output_instruction>
    Extract and format the employment history information below:
    </output_instruction>

    """

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content.strip()
    except Exception as e:
        logging.error(f"Error getting page description: {str(e)}")
        return f"Error occurred while processing the image: {str(e)}"


def generate_unique_id(name, count):
    clean_name = "".join(e for e in name if e.isalnum()).lower()
    return f"{clean_name}_{count}"


def write_to_csv(data, output_path, unique_persons):
    fieldnames = [
        "Hash ID",
        "Person ID",
        "Officer Name",
        "Officer Sex",
        "Agency Name",
        "Status",
        "Hire Date",
        "Separation Date",
        "Reason for Separation",
        "Filename",
        "Page Number"
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for page in data["messages"]:
            parsed_entries = parse_description(page["description"], page["filename"], page["page_number"])
            for entry in parsed_entries:
                hash_id = generate_hash_id(
                    entry.get("Officer Name", "Unknown"),
                    entry.get("Officer Sex", "Unknown"),
                    entry.get("Agency Name", "Unknown"),
                    entry.get("Status", "Unknown"),
                    entry.get("Hire Date", "Unknown"),
                    entry.get("Separation Date", "Unknown"),
                    entry.get("Reason for Separation", "Unknown")
                )
                person_id = generate_person_id(
                    entry.get("Officer Name", "Unknown"),
                    entry.get("Officer Sex", "Unknown")
                )
                entry_with_ids = {
                    "Hash ID": hash_id,
                    "Person ID": person_id
                }
                for field in fieldnames[2:]:  # Skip "Hash ID" and "Person ID"
                    entry_with_ids[field] = entry.get(field, "N/A")
                
                writer.writerow(entry_with_ids)
                if person_id not in unique_persons:
                    unique_persons[person_id] = []
                unique_persons[person_id].append(entry_with_ids)

def write_final_csv(unique_persons, output_path):
    fieldnames = [
        "Hash ID",
        "Person ID",
        "Officer Name",
        "Officer Sex",
        "Agency Name",
        "Status",
        "Hire Date",
        "Separation Date",
        "Reason for Separation",
        "Filename",
        "Page Number"
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for person_id, entries in unique_persons.items():
            for entry in entries:
                writer.writerow({field: entry.get(field, "N/A") for field in fieldnames})


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_persons = {}

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_data = process_pdf(pdf_path)

            json_filename = os.path.join(
                OUTPUT_DIR, os.path.splitext(filename)[0] + "_output.json"
            )
            with open(json_filename, "w") as outfile:
                json.dump(output_data, outfile, indent=2)

            csv_filename = os.path.join(
                OUTPUT_DIR, os.path.splitext(filename)[0] + "_output.csv"
            )
            write_to_csv(output_data, csv_filename, unique_persons)

            logging.info(
                f"Processed {filename} and saved results to JSON and CSV files"
            )

    final_csv_filename = os.path.join(OUTPUT_DIR, "all_officers_output.csv")
    write_final_csv(unique_persons, final_csv_filename)
    logging.info(f"Created final CSV with all unique persons: {final_csv_filename}")

if __name__ == "__main__":
    main()