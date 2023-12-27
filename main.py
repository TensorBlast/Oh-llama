import tkinter as tk    
from tkinter import ttk
from pydantic import BaseModel, Field
from typing import Optional, List, OrderedDict
from uuid import uuid4 as v4, UUID
import requests
import ast

class ChatMessage(BaseModel):
    content : str = Field(default=None, title='Message')
    role: str = Field(default=None, title='Role')

    def to_json(self):
        return self.model_dump()

class Conversation(BaseModel):
    id: UUID = Field(default=v4(), title='ID')
    messages : Optional[List[ChatMessage]] = Field(default=None, title='Messages')

    def to_json(self):
        return self.model_dump()['messages']
    
class Params(BaseModel):
    num_keep : int = Field(5)
    num_predict : int = Field(100)
    top_k : int = Field(40)
    top_p : float = Field(0.9)
    tfs_z : float = Field(0.5)
    typical_p :float = Field(0.7)
    repeat_last_n :int = Field(33)
    temperature : float = Field(0.7)
    repeat_penalty : float = Field(1)
    presence_penalty : float = Field(1.5)
    frequency_penalty : float = Field(1.0)
    mirostat : int = Field(1)
    mirostat_tau : float = Field(0.8)
    mirostat_eta : float = Field(0.6)
    penalize_newline : bool = Field(True)
    stop : Optional[List[str]] = Field(['user:'])
    numa : bool = Field(False)
    num_ctx : int = Field(1024)
    num_batch : int = Field(2)
    num_gqa : int = Field(1)
    num_gpu : int = Field(1)
    main_gpu : Optional[int] = Field(0)
    low_vram : bool = Field(False)
    f16_kv : bool = Field(True)
    vocab_only : bool = Field(False)
    use_mmap : bool = Field(True)
    use_mlock : bool = Field(False)
    embedding_only : bool = Field(False)
    rope_frequency_base : float = Field(1.1)
    rope_frequency_scale : float = Field(0.8)
    num_thread : Optional[int] = Field(8)


class OllamaChatBotGUI(tk.Tk):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title(title)
        self.params = Params()
        self.convos: List[Conversation()] = []
        self.first_convo = self.new_conversation()
        self.current_convo_id = self.first_convo.id
        self.current_conversation = self.first_convo

        self.populate_models()
        
        self.model = ":".join(self.models[0].split(':')[:-1])

        self.setupMainWindow()

    def new_conversation(self):
        convo = Conversation(id=v4(), messages=[])
        self.convos.append(convo)
        self.update()
        print(len(self.convos))
        return convo
    
    def get_conversation(self, convo_id):
        for convo in self.convos:
            if convo.id == convo_id:
                return convo
        return None

    def populate_models(self):
        url = 'http://localhost:11434/api/tags'
        response = requests.get(url)
        self.models = [x['name']+":"+x['details']['quantization_level'] for x in response.json()['models']]

    def setupMainWindow(self):
        self.mainframe = tk.Frame(self)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.mainframe.grid(row=0, column=0, sticky='nsew')

        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(1, weight=5)

        self.rightframe = tk.Frame(self.mainframe)
        self.rightframe.grid(row=0, column=1, sticky='nsew')
        self.rightframe.grid_rowconfigure(0, weight=1)
        self.rightframe.grid_columnconfigure(0, weight=1)

        # write a tkinter frame with a text area for chat messages to be appended to
        self.chatframe = tk.Frame(self.rightframe)  
        self.chatframe.grid_rowconfigure(0, weight=1)
        self.chatframe.grid_columnconfigure(0, weight=1)
        self.chatframe.grid_rowconfigure(1, weight=0)
              
        self.chatbox = tk.Text(self.chatframe)
        self.chatbox.grid(row=0, column=0, sticky='nsew')
        self.chatbox.insert(tk.END, 'Welcome to Ollama ChatBot\n')
        self.chatframe.grid(row=0, column=0, sticky='nsew')

        #Frame for the message entry box and send button
        self.entryframe = tk.Frame(self.chatframe)
        self.entryframe.grid_rowconfigure(0, weight=1)
        self.entryframe.grid_columnconfigure(0, weight=1)
        self.entry = tk.Entry(self.entryframe)
        self.sendbutton = tk.Button(self.entryframe, text='Send', command=self.send)
        self.entry.grid(row=0, column=0, sticky='nsew')
        self.sendbutton.grid(row=0, column=1, sticky='nsew')
        self.entryframe.grid(row=1, column=0, sticky='nsew')

        self.leftpanel()

    def send(self):
        # Get the message from the entry box
        message = self.entry.get()
        self.current_conversation.messages.append(ChatMessage(content=message, role='user'))
        # Clear the entry box
        self.entry.delete(0, tk.END)
        # Add the message to the chatbox
        self.chatbox.insert(tk.END, f'You: {message}\n\n')
        self.update()

        self.response = self.get_response(self.current_conversation)['message']['content']
        self.current_conversation.messages.append(ChatMessage(content=self.response, role='assistant'))

        self.chatbox.insert(tk.END, f'Bot: {self.response}\n\n')
        self.update()
        self.draw_conversations(self.leftframe)
        
    def get_response(self, conversation):
        url = 'http://localhost:11434/api/chat'

        payload = {"model": self.model, "messages": conversation.to_json(), "stream": False, "options": self.params.model_dump()}
        headers = {'content-type': 'application/json'}

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    def leftpanel(self):
        # Frame on the left side containing ollama model selection conversations
        self.leftframe = tk.Frame(self.mainframe)
        self.leftframe.grid_rowconfigure(0, weight=0)
        self.leftframe.grid_columnconfigure(0, weight=1)
        self.leftframe.grid_rowconfigure(1, weight=0)
        self.leftframe.grid_rowconfigure(2, weight=0)
        self.leftframe.grid_rowconfigure(3, weight=0)
        self.leftframe.grid_rowconfigure(4, weight=0)
        
        self.modellabel = tk.Label(self.leftframe, text='Select Model')
        self.dropdown = tk.OptionMenu(self.leftframe, tk.StringVar(value=self.models[0]), *self.models, command=self.model_selected)
        self.modellabel.grid(row=0, column=0, sticky='nsew')
        self.dropdown.grid(row=1, column=0, sticky='nsew')
        self.paramlabel = tk.Label(self.leftframe, text='Options')
        self.paramlabel.grid(row=2, column=0, sticky='nsew')
        self.parambutton = tk.Button(self.leftframe, text='Options', command=self.show_options)
        self.parambutton.grid(row=3, column=0, sticky='nsew')
        self.leftframe.grid(row=0, column=0, sticky='nsew')

        self.convolabel = tk.Label(self.leftframe, text='Conversations')
        self.convolabel.grid(row=4, column=0, sticky='nsew')

        self.draw_conversations(self.leftframe)

    def show_options(self):
        self.options_window = tk.Toplevel(self)
        self.options_window.title('Options')
        self.options_window.geometry('600x800')
        self.options_window.resizable(False, False)
        self.options_window.grab_set()
        self.options_window.focus_set()

        self.entriesCanvas = tk.Canvas(self.options_window, borderwidth=0)
        self.entriesFrame = tk.Frame(self.entriesCanvas)
        self.scrollbar = tk.Scrollbar(self.options_window, command=self.entriesCanvas.yview)
        self.entriesCanvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill='y')
        self.entriesCanvas.pack(side='left', fill='both', expand=True)
        self.entriesCanvas.create_window((0, 0), window=self.entriesFrame, anchor='nw')

        self.entriesFrame.bind('<Configure>', self.onFrameConfigure)

        self.labeloptions = []
        self.textoptions = []
        self.orderedparams = OrderedDict(self.params.model_dump())
        for i, (key, value) in enumerate(self.orderedparams.items()):
            self.labeloptions.append(tk.Label(self.entriesFrame, text=key))
            self.labeloptions[-1].grid(row=i, column=0, padx=10, pady=5)
            self.textoptions.append(tk.Entry(self.entriesFrame))
            self.textoptions[-1].insert(tk.END, value)
            self.textoptions[-1].grid(row=i, column=1, padx=10, pady=5)

        self.options_button = tk.Button(self.entriesFrame, text='Update', command=self.update_options)
        self.options_button.grid(row=len(self.orderedparams), column=0, columnspan=2, padx=10, pady=5)

    def onFrameConfigure(self, event):
        self.entriesCanvas.configure(scrollregion=self.entriesCanvas.bbox("all"))

    def update_options(self):
        for key,value in self.orderedparams.items():
            if key == 'stop':
                item = self.textoptions.pop(0).get()
                if '[' in item:
                    items = item.replace('[', '').replace(']', '').split(',')
                    self.orderedparams[key] = [x.strip() for x in items]
                else:
                    items = item.split(',')
                    if not isinstance(items, list):
                        items = [items] 
                    self.orderedparams[key] = items
                continue
            item = self.textoptions.pop(0).get()
            self.orderedparams[key] = ast.literal_eval(item)
        self.params = Params(**self.orderedparams)
        self.options_window.destroy()

    def draw_conversations(self, frame):
        convo_frame = tk.Frame(frame)
        convo_frame.grid_rowconfigure(0, weight=1)
        convo_frame.grid_columnconfigure(0, weight=1)
        convo_frame.grid_rowconfigure(1, weight=1)
        convo_frame.grid(row=5, column=0, sticky='nsew')
        new_convo_button = tk.Button(convo_frame, text='New Conversation', command=lambda: self.set_new_conversation())
        del_convo_button = tk.Button(convo_frame, text='Delete Conversation', command=lambda: self.delete_conversation())
        new_convo_button.grid(row=0, column=0, sticky='nsew')
        del_convo_button.grid(row=1, column=0, sticky='nsew')
        for i, convo in enumerate(self.convos):
            print(f"In draw loop: {convo.id}. Current convo: {self.current_convo_id}. Row {i+2}")
            if convo.id == self.current_convo_id:
                title = 'Current Conversation'
            else:
                if len(convo.messages) == 0:
                    title = 'New Conversation'
                else:
                    title = convo.messages[0].content[:20]
            
            convo_frame.grid_rowconfigure(i+2, weight=1)
            convo_button = tk.Button(convo_frame, text=title, command=lambda convo_id = convo.id: self.select_conversation(convo_id))
            convo_button.grid(row=i+2, column=0, sticky='nsew')

    def set_new_conversation(self):
        self.current_conversation= self.new_conversation()
        self.current_convo_id = self.current_conversation.id
        self.chatbox.delete('1.0', tk.END)
        self.draw_conversations(self.leftframe)
        self.update()

    def delete_conversation(self):
        if len(self.convos) == 1:
            return
        self.convos.remove(self.current_conversation)
        self.select_conversation(self.convos[0].id)
        self.draw_conversations(self.leftframe)
        self.update()

    def get_prompt_template(self, model):
        url = 'http://localhost:11434/api/show'
        payload = {"name": model}
        headers = {'content-type': 'application/json'}

        response = requests.post(url, json=payload, headers=headers)
        print(response.json()['template'])
        return response.json()['template']
    
    def select_conversation(self, convo_id):
        self.current_convo_id = convo_id
        self.current_conversation = self.get_conversation(convo_id)
        print('Selected Convo ', convo_id)
        print("Number of messages: ", len(self.current_conversation.messages))
        self.chatbox.delete('1.0', tk.END)
        for message in self.current_conversation.messages:
            if message.role == 'user':
                self.chatbox.insert(tk.END, f'You: {message.content}\n\n')
            else:
                self.chatbox.insert(tk.END, f'Bot: {message.content}\n\n')
        self.draw_conversations(self.leftframe)
        self.update()

    def model_selected(self, value):
        self.model = ":".join(value.split(':')[:-1])



if __name__ == '__main__':
    win = OllamaChatBotGUI('Ollama ChatBot')
    win.mainloop()