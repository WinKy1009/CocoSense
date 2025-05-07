import flet as ft
import json
import os
import sqlite3
import asyncio
import requests
from datetime import datetime

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

init_db()

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

    def navigate_to(route):
        page.views.clear()
        routes = {
            "Home": home_page,
            "Detection": lambda: detection_page(page),
            "Settings": settings_page,
            "Theme Preferences": theme_preferences_page,
            "History": lambda: history_page(page),
            "Help": help_page,
            "Get Started": get_started_page,
            "About": about_page
        }
        content = routes.get(route, lambda: ft.Text("Page not found", size=20))()
        if asyncio.iscoroutine(content):
            asyncio.create_task(content)
        else:
            page.views.append(content)
            page.update()

    async def get_started_page():
        loading_content = ft.Column([
            ft.Container(
                content=ft.Image(
                    src="assets/coconut.png",
                    fit=ft.ImageFit.CONTAIN
                ),
                expand=True
            ),
            ft.Text("Welcome to CocoSense", size=35, weight="bold", text_align=ft.TextAlign.CENTER, color=ft.colors.INVERSE_SURFACE)
        ], expand=True)

        page.views.append(ft.View("Get Started", [loading_content]))
        page.update()
        await asyncio.sleep(2)
        navigate_to("Home")

    def home_page():
        return ft.View("Home", [
            ft.Container(content=ft.Row([ft.Text("CocoSense", size=30, weight="bold", color="green")]), padding=ft.padding.only(top=40)),
            ft.Image(src="assets/coconut.png", width=350, height=180, fit=ft.ImageFit.FILL),
            ft.Row([
                ft.Container(
                    content=ft.Column([ft.Icon(ft.icons.SEARCH, size=50, color="black"), ft.Text("DETECTION", color=ft.colors.SURFACE)],
                    alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.GREEN, padding=20, border_radius=30,
                    on_click=lambda _: navigate_to("Detection"), expand=True),
                ft.Container(
                    content=ft.Column([ft.Icon(ft.icons.SETTINGS, size=50, color="black"), ft.Text("SETTINGS", color=ft.colors.SURFACE)],
                    alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.YELLOW, padding=20, border_radius=30,
                    on_click=lambda _: navigate_to("Settings"), expand=True)
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ft.Row([
                ft.Container(
                    content=ft.Column([ft.Icon(ft.icons.HISTORY, size=50, color="black"), ft.Text("HISTORY", color=ft.colors.SURFACE)],
                    alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.RED, padding=20, border_radius=30,
                    on_click=lambda _: navigate_to("History"), expand=True),
                ft.Container(
                    content=ft.Column([ft.Icon(ft.icons.CONTACT_SUPPORT, size=50, color="black"), ft.Text("HELP", color=ft.colors.SURFACE)],
                    alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=ft.colors.BLUE, padding=20, border_radius=30,
                    on_click=lambda _: navigate_to("Help"), expand=True)
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        ])

    def detection_page(page):
        detection_result = ft.Text("", size=18, weight=ft.FontWeight.BOLD)
        selected_image = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
        file_picker = ft.FilePicker(on_result=lambda e: process_selected_image(e.files))
        page.overlay.append(file_picker)

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
            ft.Container(content=ft.Row([
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                ft.Text("DETECTION", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),
            ft.Text("Select an image to analyze coconut ripeness.", size=20),
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.icons.SEARCH, size=100, color="black"),
                    ft.Text("Tap to Select Image", size=24, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                bgcolor="lightgreen", border_radius=50, alignment=ft.alignment.center,
                on_click=lambda _: file_picker.pick_files(allow_multiple=False), height=300
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
            ft.ListTile(leading=ft.Icon(ft.icons.PERSON), title=ft.Text("User Preferences"), on_click=lambda _: navigate_to("Theme Preferences")),
            ft.ListTile(leading=ft.Icon(ft.icons.INFO), title=ft.Text("About"), on_click=lambda _: navigate_to("About")),
        ])

    def theme_preferences_page():
        selected_theme = ft.RadioGroup(
            value=page.theme_mode,
            content=ft.Column([
                ft.Radio(value="light", label="Light Mode"),
                ft.Radio(value="dark", label="Dark Mode")
            ])
        )

        def on_theme_change(e):
            chosen_theme = selected_theme.value
            page.theme_mode = chosen_theme
            save_theme(chosen_theme)
            page.update()

        selected_theme.on_change = on_theme_change

        return ft.View("Theme Preferences", [
            ft.Container(
                content=ft.Row([
                    ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Settings")),
                    ft.Text("Theme Preferences", size=28, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.only(top=40)
            ),
            ft.Text("Choose your theme mode:", size=20),
            selected_theme
        ])

    def history_page(page):
        history_list = ft.ListView(expand=True, spacing=10)
        selected_ids = set()

        def load_history():
            history_list.controls.clear()
            selected_ids.clear()

            for record in get_history():
                record_id, image_data, result_text, date = record

                def on_checkbox_change(e, rid=record_id):
                    if e.control.value:
                        selected_ids.add(rid)
                    else:
                        selected_ids.discard(rid)

                checkbox = ft.Checkbox(on_change=lambda e, rid=record_id: on_checkbox_change(e, rid))
                delete_button = ft.IconButton(
                        ft.icons.DELETE,
                        tooltip="Delete this record",
                        on_click=lambda e, rid=record_id: (delete_history(rid), load_history())
                )

                tile = ft.Row([
                    checkbox,
                    ft.Container(
                        content=ft.ListTile(
                            title=ft.Text(result_text, size=16, weight=ft.FontWeight.BOLD),
                            subtitle=ft.Text(date, size=12, color=ft.colors.GREY),
                    ),
                        expand=True
                    ),
                    delete_button
                ])
                history_list.controls.append(tile)

            page.update()

        def delete_selected_records(e):
            for rid in list(selected_ids):
                delete_history(rid)
            load_history()

        def delete_all_records(e):
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            load_history()

        delete_buttons = ft.Row([
            ft.FilledButton("Delete Selected", on_click=delete_selected_records),
            ft.OutlinedButton("Delete All", on_click=delete_all_records)
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)

        load_history()

        return ft.View("History", [
            ft.Container(content=ft.Row([
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                ft.Text("HISTORY", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),
            delete_buttons,
            history_list
        ])


    def help_page():
        return ft.View("Contact Us", [
            ft.Container(content=ft.Row([
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Home")),
                ft.Text("CONTACT US", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),
            ft.Text("Developer 1", size=18), ft.Text("09261356527", size=18), ft.Text("regenecaguioa2002@gmail.com \n", size=18),
            ft.Text("Developer 2", size=18), ft.Text("09919368865", size=18), ft.Text("lawrencedelacruz@gmail.com", size=18)
        ])

    def about_page():
        return ft.View("About", [
            ft.Container(content=ft.Row([
                ft.IconButton(ft.icons.ARROW_BACK, on_click=lambda _: navigate_to("Settings")),
                ft.Text("ABOUT", size=30, weight=ft.FontWeight.BOLD)
            ]), padding=ft.padding.only(top=40)),
            ft.ListView([
                ft.Text("CocoSense is an advanced coconut ripeness detection app that utilizes deep learning and AI to analyze images...", size=18),
                ft.Text("With an intuitive interface and powerful image processing capabilities...", size=18),
                ft.Text("The app supports real-time image processing and advanced AI models...", size=18),
                ft.Text("CocoSense is the perfect tool for farmers, wholesalers, and consumers...", size=18),
                ft.Text("Our mission with CocoSense is to bridge the gap between tech and agriculture...", size=18),
                ft.Text("Future updates to CocoSense will include batch image analysis...", size=18)
            ], expand=True, spacing=20, padding=20)
        ])

    asyncio.run(get_started_page())

ft.app(target=main, assets_dir="assets")


