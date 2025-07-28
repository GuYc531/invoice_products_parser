# invoice_products_parser

A web-based application built with **Flask**, **OpenCV**, and **Python** for parsing and extracting structured data from invoice images (JPEG/JPG).
This project includes a simple UI, image preprocessing pipeline, and text extraction with Google vision API.

## 🚀 Features

- Upload invoice jpg/jpeg images through a user-friendly web interface
- Preprocess images for better OCR results using Google-vision-api
- Extract structured information from the table of the products details
- Download results as CSV or view them in the browser

## 🛠 Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS 
- **OCR & Parsing**: OpenCV, Google Vision API

## Algorithmic approach
1. We apply the Google-vision api for OCR given each detection with location in the image.
2. We find all the tables in the image and find the one with the most detected words in it
   - meaning the products list table
3. We find in the selected table it's vertical and horizontal lines to detect the cells inside of it
4. We compute the words location based on horizontal and vertical lines (column and row number)
5. We extract all the data into a table (dataframe)
 
## Limitations
1. Error handling very basic
2. Image must be vertically aligned (with some deviation it's okay)
3. Image must consist of products table
4. Products table must have vertical and horizontal lines
5. Would not work with any invoice type, it is an MVP :)

## Development setup

1. Clone repo:
    ```bash
    git clone https://github.com/your-username/invoice-parser.git
    cd invoice-parser
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
3. Must have files:
    ```bash
   your_project.json # your google cloud json token
   .env # must have GOOGLE_APPLICATION_CREDENTIALS=your_project.json, and FLASK_APP=app.py

   ```
4. Run the app:
    ```bash 
    flask run
    ```

5. Open in browser: http://127.0.0.1:5000/
 
## Further improvements
1. Give the user the option to adjust algorithm hyperparameters
2. Apply logic of 1 in UI 'reject' option
3. Add option to save final csv in external database
4. Add option to save locally few invoices in the same run

## Author

created by Guy Cohen.

Feel free to reach out or contribute to the project!


