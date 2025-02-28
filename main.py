import flet as ft
import json
import os
import torch
import numpy as np
import cv2
import sqlite3
from ultralytics import YOLO
from PIL import Image
import io
import base64
from datetime import datetime

model = YOLO("runs/detect/train3/weights/best.pt")

CLASS_NAMES = {0: "Old Coconut", 1: "Unripe Coconut", 2: "Ripe Coconut"}

DB_NAME = "coco_history.db"


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

def detect_objects(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    results = model(image)
    detection_text = ""
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASS_NAMES.get(int(cls), 'Unknown')} ({conf:.2f})"
            detection_text += f"{label}\n"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    _, encoded_image = cv2.imencode(".png", image)
    base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    # Save detection result to history
    save_to_history(base64_image, detection_text)

    return f"data:image/png;base64,{base64_image}", detection_text

def main(page: ft.Page):
    page.title = "CocoSense - Coconut Ripeness Detection"
    page.theme_mode = load_theme()  # Load the saved theme
    page.window.maximizable = True
    page.window.height = 700
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

    def get_started_page():
        return ft.Column([
            ft.Container(
                content=ft.Image(
                    src="assets/coconut.png",
                    width=page.window.width,
                    height=page.window.height,
                    fit=ft.ImageFit.CONTAIN
                ),
                expand=True
            ),
            ft.Container(
                content=ft.Column([
                    ft.Stack([
                        ft.Text("Welcome to CocoSense", size=35, weight="bold", text_align=ft.TextAlign.CENTER, color=ft.colors.INVERSE_SURFACE)
                    ]),
                    ft.Container(
                        content=ft.ElevatedButton(
                            "Get Started",
                            on_click=lambda _: navigate_to("Home"),
                            style=ft.ButtonStyle(bgcolor="green", color="white", padding=30)
                        ),
                        alignment=ft.alignment.bottom_center,
                        padding=50
                    )
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20),
                expand=True,
                padding=40,
            )
        ], expand=True)

    def home_page():
        return ft.View("Home", [
            ft.Container(
                content=ft.Row([
                    ft.Text("CocoSense", size=30, weight="bold", color="green")
                ], alignment=ft.MainAxisAlignment.START),
                padding=ft.padding.only(top=40)
            ),
            ft.Image(
                src="assets/coconut.png",  # Use local file
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
                    bgcolor="lightgreen",
                    padding=20,
                    border_radius=10,
                    on_click=lambda _: navigate_to("Detection"),
                    expand=True
                ),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.HISTORY, size=50, color="black"),
                        ft.Text("HISTORY",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor="lightblue",
                    padding=20,
                    border_radius=10,
                    on_click=lambda _: navigate_to("History"),
                    expand=True
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.SETTINGS, size=50, color="black"),
                        ft.Text("SETTINGS",color=ft.colors.SURFACE)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor="yellow",
                    padding=20,
                    border_radius=10,
                    on_click=lambda _: navigate_to("Settings"),
                    expand=True
                )
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        ])

    def detection_page(page):
        # Remove the image display since we only want to show the text result.
        detection_result = ft.Text("", size=18, weight=ft.FontWeight.BOLD)

        def process_selected_image(files):
            if files:
                file_path = files[0].path
                with open(file_path, "rb") as f:
                    image_data = f.read()
                processed_image_base64, result_text = detect_objects(image_data)
                detection_result.value = result_text
                dialog.open = True
                page.update()

        file_picker = ft.FilePicker(on_result=lambda e: process_selected_image(e.files))
        page.overlay.append(file_picker)

        dialog = ft.AlertDialog(
            title=ft.Text("Detection Result"),
            content=ft.Column([
                detection_result  # Just show the detection result here.
            ], alignment=ft.MainAxisAlignment.CENTER),
            actions=[
                ft.TextButton("OK", on_click=lambda _: setattr(dialog, 'open', False) or page.update())
            ]
        )

        page.overlay.append(dialog)

        return ft.View("Detection", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                    ft.Text("DETECTION", size=30, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            ft.Text("Select an image to analyze coconut ripeness.", size=20),
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.icons.SEARCH, size=80, color="black"),
                    ft.Text("Tap to Select Image", size=24, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                bgcolor="lightgreen",
                border_radius=50,
                alignment=ft.alignment.center,
                on_click=lambda _: file_picker.pick_files(allow_multiple=False),
                height=300,
            )
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
            ft.ListTile(leading=ft.Icon(ft.icons.CONTACT_SUPPORT), title=ft.Text("Contact Us"), on_click=lambda _: navigate_to("Help")),
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
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Settings")),
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

    # Start the app on the Get Started screen
    page.views.append(ft.View("Get Started", [get_started_page()]))
    page.update()

ft.app(target=main, assets_dir="assets")

