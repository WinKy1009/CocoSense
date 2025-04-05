import flet as ft
import json
import os
import sqlite3
import asyncio
import requests
from datetime import datetime
import cv2
import base64
import threading

DB_NAME = "coco_history.db"
API_URL = "https://Werniverse-CocoSense.hf.space/predict/"
def load_theme():
    if os.path.exists("theme.json"):
        with open("theme.json", "r") as file:
            return json.load(file).get("theme_mode", "light")
    return "light"

def save_theme(theme_mode):
    with open("theme.json", "w") as file:
        json.dump({"theme_mode": theme_mode}, file)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            result TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()  # Ensure DB is initialized

def save_to_history(image_data, result_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("INSERT INTO history (image, result, date) VALUES (?, ?, ?)", 
                   (image_data, result_text, date_now))
    
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, image, result, date FROM history ORDER BY id DESC")
    data = cursor.fetchall()
    
    conn.close()
    return data

def delete_history(record_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM history WHERE id=?", (record_id,))
    
    conn.commit()
    conn.close()

def detect_objects(image_path):
    API_URL = "https://Werniverse-CocoSense.hf.space/predict/"

    
    with open(image_path, "rb") as file:
        response = requests.post(API_URL, files={"file": file})

    if response.status_code == 200:
        result = response.json()
        return f"Predicted: {result['class']} (Confidence: {result['confidence']:.2f}%)"
    else:
        return "Error processing image"
    
def main(page: ft.Page):
    page.title = "CocoSense - Coconut Ripeness Detection"
    page.theme_mode = load_theme()
    page.window.maximizable = True
    page.window.height = 650
    page.window.width = 350
    page.window.resizable = True


    def toggle_theme():
        new_theme = "dark" if page.theme_mode == "light" else "light"
        page.theme_mode = new_theme
        save_theme(new_theme)
        page.update()

    def navigate_to(route):
        page.views.clear()
        routes = {
            "Home": home_page,
            "Detection": lambda: detection_page(page),
            "Settings": settings_page,
            "History": lambda: history_page(page),  # Pass page argument
            "Help": help_page,
            "Get Started": get_started_page,
            "About": about_page
        }
        content = routes.get(route, lambda: ft.Text("Page not found", size=20))()
        page.views.append(content)
        page.update()

    async def get_started_page(page):
        loading_content = ft.Column([
            ft.Container(
                content=ft.Image(
                    src="assets/coconut.png",
                    fit=ft.ImageFit.CONTAIN
                ),
                expand=True
            ),
            ft.Stack([
                ft.Text("Welcome to CocoSense", size=35, weight="bold", text_align=ft.TextAlign.CENTER, color=ft.colors.INVERSE_SURFACE)
            ])
        ], expand=True)
        
        page.views.append(ft.View("Get Started", [loading_content]))
        page.update()

        await asyncio.sleep(2)
            
        navigate_to("Home")

    def home_page():
        return ft.View("Home", [
            ft.Container(
                content=ft.Row([
                    ft.Text("CocoSense", size=30, weight="bold", color="green")
                ], alignment=ft.MainAxisAlignment.START),
                padding=ft.padding.only(top=40)
            ),
            ft.Image(
                src="assets/coconut.png",
                width=350,
                height=180,
                fit=ft.ImageFit.FILL
            ),
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.SEARCH, size=50, color="black"),
                        ft.Text("DETECTION",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.GREEN,
                    padding=20,
                    border_radius=30,
                    on_click=lambda _: navigate_to("Detection"),
                    expand=True
                ),

                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.SETTINGS, size=50, color="black"),
                        ft.Text("SETTINGS",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.YELLOW,
                    padding=20,
                    border_radius=30,
                    on_click=lambda _: navigate_to("Settings"),
                    expand=True
                )

            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.HISTORY, size=50, color="black"),
                        ft.Text("HISTORY",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.RED,
                    padding=20,
                    border_radius=30,
                    on_click=lambda _: navigate_to("History"),
                    expand=True
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.CONTACT_SUPPORT, size=50, color="black"),
                        ft.Text("HELP",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.BLUE,
                    padding=20,
                    border_radius=30,
                    on_click=lambda _: navigate_to("Help"),
                    expand=True
                )
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        ])

    def detection_page(page):
        detection_result = ft.Text("", size=18, weight=ft.FontWeight.BOLD)
        selected_image = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
        file_picker = ft.FilePicker(on_result=lambda e: process_selected_image(e.files))
        page.overlay.append(file_picker)

        # Function to navigate to the camera page
        def open_camera_page():
            page.views.append(camera_page(page))
            page.update()

        def process_selected_image(files):
            if files:
                file_path = files[0].path
                result_text = detect_objects(file_path)
                detection_result.value = result_text
                selected_image.src = file_path
                with open(file_path, "rb") as f:
                    save_to_history(f.read(), result_text)
                dialog.open = True
                page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Detection Result"),
            content=ft.Column([selected_image, detection_result], expand=True),
            actions=[ft.TextButton("OK", on_click=lambda _: setattr(dialog, 'open', False) or page.update())]
        )
        page.overlay.append(dialog)

        return ft.View("Detection", [
            ft.Container(content=ft.Row([  # Navigation bar with back button
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                ft.Text("DETECTION", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),

            ft.Text("Capture an image and analyze coconut ripeness.", size=20),

            ft.Container(  # Select image from device
                content=ft.Column([
                    ft.Icon(ft.icons.IMAGE, size=100, color="black"),
                    ft.Text("Tap to Select Image", size=24, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                bgcolor="lightgreen",
                border_radius=50,
                alignment=ft.alignment.center,
                on_click=lambda _: file_picker.pick_files(allow_multiple=False),
                height=200,
            ),

            # Button to open the camera for capturing image
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.icons.CAMERA_ALT, size=100, color="black"),
                    ft.Text("Capture Image", size=24, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                bgcolor="lightblue",
                border_radius=50,
                alignment=ft.alignment.center,
                on_click=lambda _: open_camera_page(),
                height=200,
            )
        ])

    def camera_page(page):
        # Initially set the image with an empty source to avoid errors
        capture_image_view = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)

        detection_result = ft.Text("", size=18, weight=ft.FontWeight.BOLD)

        # Define cap globally so it can be closed later
        cap = None

        # Function to capture a frame from the camera and update the image
        def capture_frame():
            nonlocal cap
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                detection_result.value = "Error: Camera not found"
                page.update()
                return
            
            # Capture frames indefinitely
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert the captured frame to base64 for displaying in Flet Image component
                    _, buffer = cv2.imencode('.jpg', frame)
                    if _:
                        encoded_image = base64.b64encode(buffer).decode("utf-8")
                        # Update the image view with the new frame
                        capture_image_view.src_base64 = encoded_image  # Just the base64 part
                        page.update()

        # Run the capture_frame function in a separate thread
        def start_camera_thread():
            threading.Thread(target=capture_frame, daemon=True).start()

        # Function to capture and process a single frame (when the user presses Capture button)
        def capture_and_process():
            nonlocal cap
            if cap is not None:
                cap.release()  # Stop the camera feed after capturing the image

            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): 
                detection_result.value = "Error: Camera not found"
                page.update()
                return

            ret, frame = cap.read() 
            cap.release()

            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)

                # Convert the captured image to base64 for displaying in Flet Image component
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                # Set the image src_base64 after conversion (without the "data:image/jpeg;base64," prefix)
                capture_image_view.src_base64 = encoded_image  # Just the base64 part

                result_text = detect_objects(image_path)  # Analyze image
                save_to_history(open(image_path, "rb").read(), result_text)  # Save to history
                detection_result.value = result_text

                page.update()
            else:
                detection_result.value = "Error capturing image."
                page.update()

        # Start the camera thread when the page is loaded
        start_camera_thread()

        return ft.View("Camera", [
            ft.Container(content=ft.Row([ 
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Detection")),
                ft.Text("CAPTURE IMAGE", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),

            ft.Container( 
                content=ft.Text("Live camera feed below. Tap the button below to capture an image.", size=18),
                padding=20
            ),

            capture_image_view,  # Display the live camera feed

            ft.Container(
                content=ft.ElevatedButton("Capture", icon=ft.icons.CAMERA, on_click=lambda _: capture_and_process(), width=200, height=80, expand=True),
                padding=20,
                alignment=ft.alignment.center
            ),

            detection_result
        ])

    def settings_page():
        return ft.View("Settings", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                    ft.Text("SETTINGS", size=30, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            ft.ListTile(leading=ft.Icon(ft.icons.PERSON), title=ft.Text("User Preferences"), on_click=lambda _: toggle_theme()),
            ft.ListTile(leading=ft.Icon(ft.icons.INFO), title=ft.Text("About"), on_click=lambda _: navigate_to("About")),
        ])
    
    def history_page(page):
        history_list = ft.ListView(expand=True, spacing=10)

        def load_history():
            history_list.controls.clear()
            for record in get_history():
                record_id, image_data, result_text, date = record  # Ensure the order matches the database query
                
                def delete_record(event, rid=record_id):
                    delete_history(rid)
                    load_history()

                # Append the detected results and date properly
                history_list.controls.append(
                    ft.ListTile(
                        title=ft.Text(result_text, size=16, weight=ft.FontWeight.BOLD),
                        subtitle=ft.Text(date, size=12, color=ft.colors.GREY),
                        trailing=ft.IconButton(ft.icons.DELETE, on_click=delete_record)
                    )
                )
            page.update()

        load_history()

        return ft.View("History", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                    ft.Text("HISTORY", size=30, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            history_list
    ])

    def help_page():

        return ft.View("Contact Us", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                    ft.Text("CONTACT US", size=30, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            ft.Text("Developer 1", size=18, selectable=False),
            ft.Text("09261356527", size=18, selectable=False),
            ft.Text("regenecaguioa2002@gmail.com \n", size=18, selectable=False),
            ft.Text("Developer 2", size=18, selectable=False),
            ft.Text("09919368865", size=18, selectable=False),
            ft.Text("jumalyntemporal@gmail.com", size=18, selectable=False),
        ])

    def about_page():
        return ft.View("About", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Settings")),
                    ft.Text("ABOUT", size=30, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            ft.ListView([
                ft.Text("CocoSense is an advanced coconut ripeness detection app that utilizes AI to analyze images and determine the ripeness stage of coconuts. The app is designed to assist farmers and consumers in making informed decisions about coconut harvesting and purchasing.", size=18, selectable=False),
                ft.Text("With an intuitive interface and powerful image processing capabilities, CocoSense provides quick and accurate results. Whether you're in agriculture, food processing, or just an enthusiast, this app simplifies the coconut selection process with cutting-edge AI technology.", size=18, selectable=False),
                ft.Text("The app supports real-time image processing and advanced AI models trained specifically for coconut ripeness classification. Users can simply upload a coconut image, and the app will instantly provide an accurate ripeness prediction.", size=18, selectable=True),
                ft.Text("CocoSense is the perfect tool for farmers looking to optimize their harvest, wholesalers ensuring product quality, and consumers selecting the best coconuts at the market. By leveraging AI and deep learning, the app eliminates guesswork and enhances productivity in coconut-related industries.", size=18, selectable=False),
                ft.Text("Our mission with CocoSense is to bridge the gap between technology and agriculture, offering a smart and efficient way to assess coconut ripeness. By reducing waste and improving decision-making, we aim to create a more sustainable future for coconut farming.", size=18, selectable=False),
                ft.Text("Future updates to CocoSense will include additional features like batch image analysis, predictive insights for farmers, and integration with smart farming solutions. Stay tuned for more innovations that make coconut farming and purchasing smarter and easier!", size=18, selectable=False)
            ], expand=True, spacing=20, padding=20)
        ])

    asyncio.run(get_started_page(page)) 

ft.app(target=main, assets_dir="assets")
