import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import os
from rag_llama_index import train_model, get_response

class ChatboxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Sample")
        self.root.geometry("500x600")

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame to hold the directory entry and browse button
        directory_frame = tk.Frame(self.root)
        directory_frame.pack(pady=10)

        # Directory input label
        directory_label = tk.Label(directory_frame, text="Directory Path:")
        directory_label.pack(side=tk.LEFT, padx=5)

        # Directory input field
        self.directory_entry = tk.Entry(directory_frame, width=40)
        self.directory_entry.pack(side=tk.LEFT, padx=5)

        # Browse button to select directory
        browse_button = tk.Button(directory_frame, text="Browse", command=self.browse_directory)
        browse_button.pack(side=tk.LEFT, padx=5)

        # Chat window (scrollable)
        self.chat_window = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=60)
        self.chat_window.pack(pady=10)

        # User input field
        self.user_input = tk.Entry(self.root, width=50)
        self.user_input.pack(pady=5)

        # Send button
        send_button = tk.Button(self.root, text="Send", command=self.send_message)
        send_button.pack()

        # Bind the Enter key to sending the message
        self.root.bind('<Return>', lambda event: self.send_message())

    def send_message(self):
        user_message = self.user_input.get()
        if user_message.strip():
            self.chat_window.config(state=tk.NORMAL)
            self.chat_window.insert(tk.END, "You: " + user_message + "\n")
            self.chat_window.insert(tk.END, "System: " + get_response(self.index, user_message) + "\n\n")
            self.chat_window.config(state=tk.DISABLED)
            self.chat_window.yview(tk.END)
            self.user_input.delete(0, tk.END)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.directory_entry.delete(0, tk.END)
            self.directory_entry.insert(0, directory)
            absolute_path = os.path.abspath(directory)
            self.index = train_model(absolute_path)
            self.clear_chat()
            messagebox.showinfo("Model Training", f"Model trained on files in {absolute_path} directory")

    def clear_chat(self):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete(1.0, tk.END)
        self.chat_window.config(state=tk.DISABLED)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatboxApp(root)
    root.mainloop()
