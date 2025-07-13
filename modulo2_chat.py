import os
import google.generativeai as genai
import sqlite3
import datetime
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk # Import ttk for themed widgets
import threading

# Importar la función de predicción del módulo de visión
try:
    from modulo1_vision import predecir_emocion_y_persona
except ImportError:
    messagebox.showerror("Error de Importación",
                          "No se pudo importar 'predecir_emocion_y_persona' del archivo 'modulo1_vision.py'. "
                          "Asegúrate de que el archivo existe y el Módulo 1 se ejecutó para generar 'mejor_modelo.h5'.")
    exit()

# Configura la clave API del LLM
API_KEY = "AIzaSyBeD-YR_lw9pyIahzsi-6Cv4-D-Sh6L-E8"
genai.configure(api_key=API_KEY)

# Inicializa el Modelo Generativo (LLM)
model_llm = genai.GenerativeModel('gemini-pro')

# --- Base de Datos para Conversaciones del Chat ---
DB_NAME = 'chat_history.db'

def setup_database():
    """Configura la tabla de conversaciones en la base de datos."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            speaker TEXT NOT NULL,
            message TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_message(user_id, speaker, message):
    """Guarda un mensaje en la base de datos."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("INSERT INTO conversations (user_id, timestamp, speaker, message) VALUES (?, ?, ?, ?)",
                   (user_id, timestamp, speaker, message))
    conn.commit()
    conn.close()

def get_conversation_history(user_id, limit=5):
    """Recupera el historial de conversación para un usuario dado."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT speaker, message FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    history = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "parts": [row[1]]} for row in reversed(history)]

# --- Configuración de Prompts para diferentes usuarios y emociones ---
user_prompts = {
    'personaA': {
        'initial': "Eres un agente de chat amigable y entusiasta. Estás hablando con Persona A. Haz tus respuestas atractivas y positivas.",
        'alegre': "¡Persona A está feliz! Responde con mensajes alegres y alentadores.",
        'triste': "Persona A parece triste. Ofrece consuelo e intenta animarle suavemente.",
        'pensativo': "Persona A está pensativa. Participa en conversaciones reflexivas e interesantes.",
        'con_ira': "Persona A parece enfadada. Responde con calma e intenta calmar la situación.",
        'cansado': "Persona A parece cansada. Ofrece palabras de consuelo y sugiere descanso.",
        'sorprendido': "¡Persona A está sorprendida! Expresa asombro y curiosidad.",
        'riendo': "¡Persona A se está riendo! Comparte su alegría con respuestas juguetonas y desenfadadas."
    },
    'personaB': {
        'initial': "Eres un agente de chat formal e informativo. Estás hablando con Persona B. Proporciona información concisa y útil.",
        'alegre': "¡Persona B está feliz! Mantén un tono educado pero positivo.",
        'triste': "Persona B parece triste. Ofrece apoyo práctico o un silencio empático.",
        'pensativo': "Persona B está pensativa. Proporciona información fáctica o haz preguntas que inviten a la reflexión.",
        'con_ira': "Persona B parece enfadada. Aborda sus preocupaciones profesionalmente y busca resolver los problemas.",
        'cansado': "Persona B parece cansada. Reconoce su estado y ofrece información concisa.",
        'sorprendido': "¡Persona B está sorprendida! Proporciona contexto o explicaciones.",
        'riendo': "¡Persona B se está riendo! Reconoce su diversión breve y cortésmente."
    }
}

# Variables globales para el estado del chat
current_user = None
current_emotion = None
chat_session = None
chat_window = None
chat_display = None
user_entry = None
send_button = None
initial_photo_path_entry = None
initial_photo_button = None
process_initial_button = None
update_emotion_photo_entry = None
update_emotion_button = None

# --- Funciones de la GUI ---
def send_message_gui():
    global current_user, current_emotion, chat_session

    user_input = user_entry.get().strip()
    user_entry.delete(0, tk.END)

    if not user_input:
        return

    display_message("Tú: " + user_input, "user_msg")
    save_message(current_user, "user", user_input)

    user_entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)

    threading.Thread(target=process_chatbot_response, args=(user_input,)).start()

def process_chatbot_response(user_input):
    global current_user, current_emotion, chat_session

    try:
        history_for_llm = get_conversation_history(current_user, limit=5)

        full_prompt_for_llm = (
            f"Contexto actual (usuario: {current_user}, emoción: {current_emotion}): "
            f"{user_prompts[current_user][current_emotion]}\n"
            f"Historial reciente: {history_for_llm}\n"
            f"Mensaje del usuario: {user_input}"
        )

        response_from_llm = chat_session.send_message(full_prompt_for_llm)
        bot_response = response_from_llm.text
        save_message(current_user, "model", bot_response)
        display_message("Agente: " + bot_response, "agent_msg")

    except Exception as e:
        display_message(f"Error al comunicarse con el LLM: {e}", "error_msg")
        display_message("Por favor, asegúrate de que la API key es válida y tienes conexión a internet.", "error_msg")
    finally:
        user_entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)

def display_message(message, tag="default"):
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, message + "\n", tag)
    chat_display.config(state=tk.DISABLED)
    chat_display.yview(tk.END)

def select_initial_photo():
    file_path = filedialog.askopenfilename(
        title="Selecciona tu foto inicial",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        initial_photo_path_entry.delete(0, tk.END)
        initial_photo_path_entry.insert(0, file_path)

def process_initial_photo():
    global current_user, current_emotion, chat_session

    photo_path = initial_photo_path_entry.get().strip()
    if not photo_path:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una foto inicial.")
        return

    if not os.path.exists(photo_path):
        messagebox.showerror("Error", f"La ruta de la foto no existe: {photo_path}")
        return

    display_message("Analizando la foto para identificar al usuario y su emoción inicial...", "info_msg")
    initial_photo_button.config(state=tk.DISABLED)
    initial_photo_path_entry.config(state=tk.DISABLED)
    process_initial_button.config(state=tk.DISABLED)

    threading.Thread(target=run_initial_detection, args=(photo_path,)).start()

def run_initial_detection(photo_path):
    global current_user, current_emotion, chat_session

    person, emotion = predecir_emocion_y_persona(photo_path)

    if person and emotion:
        current_user = person
        current_emotion = emotion
        display_message(f"¡Hola {current_user}! Te veo {current_emotion}. 😊", "system_msg")

        system_prompt = user_prompts[current_user]['initial'] + " " + user_prompts[current_user][current_emotion]
        chat_history_for_session = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Ok, estoy listo para conversar."]}
        ]
        chat_session = model_llm.start_chat(history=chat_history_for_session)
        display_message("Agente: Ok, estoy listo para conversar.", "agent_msg")
        save_message(current_user, "user", system_prompt)
        save_message(current_user, "model", "Ok, estoy listo para conversar.")

        user_entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)
        update_emotion_button.config(state=tk.NORMAL)
        update_emotion_photo_entry.config(state=tk.NORMAL)
        display_message("Puedes chatear ahora o actualizar tu emoción con una nueva foto.", "info_msg")

    else:
        messagebox.showerror("Error de Detección", "No se pudo identificar al usuario en la foto. Por favor, intenta con otra foto.")
        display_message("No se pudo identificar al usuario en la foto.", "error_msg")
        initial_photo_button.config(state=tk.NORMAL)
        initial_photo_path_entry.config(state=tk.NORMAL)
        process_initial_button.config(state=tk.NORMAL)

def select_update_emotion_photo():
    file_path = filedialog.askopenfilename(
        title="Selecciona una foto para actualizar tu emoción",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        update_emotion_photo_entry.delete(0, tk.END)
        update_emotion_photo_entry.insert(0, file_path)
        process_update_emotion_photo()

def process_update_emotion_photo():
    global current_user, current_emotion, chat_session

    if current_user is None:
        messagebox.showwarning("Advertencia", "Primero debes identificar al usuario con la foto inicial.")
        return

    photo_path = update_emotion_photo_entry.get().strip()
    if not photo_path:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una foto para actualizar la emoción.")
        return

    if not os.path.exists(photo_path):
        messagebox.showerror("Error", f"La ruta de la foto no existe: {photo_path}")
        return

    display_message("Analizando la nueva foto para actualizar tu emoción...", "info_msg")
    update_emotion_button.config(state=tk.DISABLED)
    update_emotion_photo_entry.config(state=tk.DISABLED)

    threading.Thread(target=run_emotion_update, args=(photo_path,)).start()

def run_emotion_update(photo_path):
    global current_user, current_emotion, chat_session

    _, new_emotion = predecir_emocion_y_persona(photo_path)

    if new_emotion:
        if new_emotion != current_emotion:
            display_message(f"¡Oh, ahora te veo {new_emotion}! Mi tono de conversación se adaptará. 😊", "system_msg")
            current_emotion = new_emotion
            emotion_update_message = f"El usuario {current_user} ahora parece estar {current_emotion}. Adapta tu respuesta a esta emoción."
            save_message(current_user, "user", emotion_update_message)
            try:
                response_from_llm = chat_session.send_message(emotion_update_message)
                save_message(current_user, "model", response_from_llm.text)
                display_message("Agente: " + response_from_llm.text, "agent_msg")
            except Exception as e:
                display_message(f"Error al enviar mensaje al LLM: {e}", "error_msg")
                display_message("Por favor, asegúrate de que la API key es válida y tienes conexión a internet.", "error_msg")
        else:
            display_message(f"Sigues pareciendo {current_emotion}. No hay cambio en mi tono de conversación.", "info_msg")
    else:
        display_message("No se pudo detectar una emoción en la nueva foto.", "error_msg")
    
    update_emotion_button.config(state=tk.NORMAL)
    update_emotion_photo_entry.config(state=tk.NORMAL)

def create_gui():
    global chat_window, chat_display, user_entry, send_button
    global initial_photo_path_entry, initial_photo_button, process_initial_button
    global update_emotion_photo_entry, update_emotion_button

    setup_database()

    chat_window = tk.Tk()
    chat_window.title("🤖 Agente de Chat con Visión")
    chat_window.geometry("750x800")
    chat_window.resizable(False, False)
    chat_window.config(bg="#f0f2f5") # Light gray background

    # Configure a style for ttk widgets
    style = ttk.Style()
    style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'
    style.configure('TFrame', background='#f0f2f5')
    style.configure('TLabelFrame', background='#f0f2f5', foreground='#333333', font=('Arial', 11, 'bold'))
    style.configure('TLabel', background='#f0f2f5', foreground='#333333')
    style.configure('TButton', font=('Arial', 10, 'bold'), background='#007bff', foreground='white', relief='flat')
    style.map('TButton', background=[('active', '#0056b3')])
    style.configure('TEntry', font=('Arial', 10))

    # --- Main Title ---
    title_label = ttk.Label(chat_window, text="Agente de Conversación Inteligente",
                            font=("Arial", 16, "bold"), foreground="#007bff")
    title_label.pack(pady=15)

    # --- Frame para la carga de la foto inicial ---
    initial_photo_frame = ttk.LabelFrame(chat_window, text="📸 Identificación Inicial", padding=(15, 10))
    initial_photo_frame.pack(pady=5, padx=20, fill="x")

    tk.Label(initial_photo_frame, text="Ruta de la foto inicial:", font=('Arial', 10)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    initial_photo_path_entry = ttk.Entry(initial_photo_frame, width=50)
    initial_photo_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    initial_photo_button = ttk.Button(initial_photo_frame, text="Buscar Foto...", command=select_initial_photo)
    initial_photo_button.grid(row=0, column=2, padx=5, pady=5)

    process_initial_button = ttk.Button(initial_photo_frame, text="Procesar Foto", command=process_initial_photo)
    process_initial_button.grid(row=0, column=3, padx=5, pady=5)
    initial_photo_frame.grid_columnconfigure(1, weight=1) # Allow entry to expand

    # --- Área de visualización del chat ---
    chat_display_frame = ttk.Frame(chat_window, relief="solid", borderwidth=1, padding=10)
    chat_display_frame.pack(pady=10, padx=20, fill="both", expand=True)
    chat_display_frame.config(style='ChatDisplay.TFrame') # Apply a custom style if needed
    style.configure('ChatDisplay.TFrame', background='white', bordercolor='#cccccc')

    chat_display = scrolledtext.ScrolledText(chat_display_frame, wrap=tk.WORD, state=tk.DISABLED,
                                            font=("Helvetica", 11), bg="#ffffff", fg="#333333", relief="flat", padx=10, pady=10)
    chat_display.pack(fill="both", expand=True)

    # Configurar tags para colores de mensaje (using more muted, professional colors)
    chat_display.tag_config("user_msg", foreground="#0056b3")      # Darker blue for user
    chat_display.tag_config("agent_msg", foreground="#28a745")     # Green for agent
    chat_display.tag_config("error_msg", foreground="#dc3545")     # Red for errors
    chat_display.tag_config("info_msg", foreground="#6c757d")      # Gray for info
    chat_display.tag_config("system_msg", foreground="#6f42c1")    # Purple for system/emotion updates


    # --- Frame para la entrada de mensaje y el botón de enviar ---
    input_frame = ttk.Frame(chat_window, padding=(10, 5))
    input_frame.pack(pady=5, padx=20, fill="x")

    user_entry = ttk.Entry(input_frame, font=("Helvetica", 11))
    user_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=5, ipady=3) # Add internal padding
    user_entry.bind("<Return>", lambda event=None: send_message_gui())

    send_button = ttk.Button(input_frame, text="Enviar 🚀", command=send_message_gui)
    send_button.pack(side=tk.RIGHT, padx=5)

    # --- Frame para actualizar la emoción ---
    update_emotion_frame = ttk.LabelFrame(chat_window, text="🔄 Actualizar Emoción", padding=(15, 10))
    update_emotion_frame.pack(pady=10, padx=20, fill="x")

    tk.Label(update_emotion_frame, text="Nueva foto:", font=('Arial', 10)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    update_emotion_photo_entry = ttk.Entry(update_emotion_frame, width=50)
    update_emotion_photo_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    update_emotion_button = ttk.Button(update_emotion_frame, text="Detectar Emoción", command=select_update_emotion_photo)
    update_emotion_button.grid(row=0, column=2, padx=5, pady=5)
    update_emotion_frame.grid_columnconfigure(1, weight=1) # Allow entry to expand

    # Inicialmente deshabilitar elementos del chat hasta que el usuario sea identificado
    user_entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)
    update_emotion_photo_entry.config(state=tk.DISABLED)
    update_emotion_button.config(state=tk.DISABLED)

    display_message("¡Bienvenido al Agente de Chat con Visión! 👋", "info_msg")
    display_message("Para empezar, selecciona y 'Procesar Foto' para identificarte.", "info_msg")

    chat_window.mainloop()

if __name__ == '__main__':
    create_gui()