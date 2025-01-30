import customtkinter as ctk
from tkinter import filedialog, StringVar
from rag_llama_index import train_model, get_response
import os
from QAForAbstra import ask

class SampleDialog(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sample Dialog")
        self.geometry("600x700")  # Fixed size
        self.resizable(False, False)  # Make the size fixed

        # Directory input and load button
        self.dir_label = ctk.CTkLabel(self, text="Select Directory:")
        self.dir_label.pack(pady=10)

        self.dir_button = ctk.CTkButton(self, text="Browse", command=self.load_directory)
        self.dir_button.pack(pady=10)

        self.message_label = ctk.CTkLabel(self, text="", text_color="green")
        self.message_label.pack(pady=5)

        # Dropdown Menu
        self.option_var = StringVar(value="RAG")
        self.dropdown = ctk.CTkOptionMenu(self, variable=self.option_var, values=["RAG", "Abstra Documents QA"])
        self.dropdown.pack(pady=10)

        # Chatbox UI
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.chat_input = ctk.CTkEntry(self, placeholder_text="Type a message...")
        self.chat_input.pack(fill='x', padx=10, pady=5)
        self.chat_input.bind("<Return>", self.send_message)
        self.chat_input.focus()

    def load_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.message_label.configure(text=f"Directory Loaded: {directory}", text_color="green")
            absolute_path = os.path.abspath(directory)
            self.index = train_model(absolute_path)
        else:
            self.message_label.configure(text="No directory selected", text_color="red")
        
        for widget in self.chat_frame.winfo_children():
            widget.pack_forget()
        self.chat_input.focus()

    def send_message(self, event=None):
        user_message = self.chat_input.get()
        if user_message.strip():
            self.display_message(user_message, align="right", bg_color="#D1E8FF")
            self.chat_input.delete(0, 'end')
            # Simulate system response (without logic)
            if(self.dropdown.get() == "RAG"):
                print("RAG Selected")
                response = get_response(self.index, user_message)
            elif(self.dropdown.get() == "Abstra Documents QA"):
                print("Abstra Embed Selected")
                response = ask(user_message)
            self.display_message(response, align="left", bg_color="#E8E8E8")

    def display_message(self, message, align="left", bg_color="#E8E8E8"):
        # Adjust textbox height dynamically based on content
        line_count = message.count('\n') + 1
        char_width = 40
        wrapped_lines = sum([len(line) // char_width + 1 for line in message.split('\n')])
        height = max(40, 13 * max(line_count, wrapped_lines))
        print("WL: ", wrapped_lines)
        print("LC: ", line_count)
        print("Height: ", height)

        # Create a new textbox for each message
        textbox = ctk.CTkTextbox(self.chat_frame, height=height, width=450, wrap='word', fg_color=bg_color, text_color="black")
        textbox.insert('0.0', message)
        textbox.configure(state='disabled', font=("Arial", 12, "bold"))

        # Align based on sender
        if align == "right":
            textbox.pack(padx=10, pady=5, anchor='e')  # Align to right
        else:
            textbox.pack(padx=10, pady=5, anchor='w')  # Align to left

if __name__ == "__main__":
    app = SampleDialog()
    app.mainloop()
