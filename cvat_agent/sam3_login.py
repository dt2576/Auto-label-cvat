"""Authenticate locally with a hidden token prompt, without device-code expiry."""
import getpass
import os


def clean_token(value):
    value = value.strip().strip('\"\'`').strip()
    if value.lower().startswith('bearer '):
        value = value[7:].strip()
    return value


def prompt_token():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        while True:
            value = clean_token(getpass.getpass('Read token (right-click or Shift+Insert to paste): '))
            if value.startswith('hf_'):
                return value
            print('Copy the complete token beginning with hf_. Do not paste a URL or device code.')
    root.title('Hugging Face - SAM3 login')
    root.geometry('580x230')
    root.resizable(False, False)
    root.attributes('-topmost', True)
    tk.Label(root, text='Paste your Hugging Face Read token (begins with hf_).',
             padx=16, pady=12).pack(anchor='w')
    entry = tk.Entry(root, show='*', width=66)
    entry.pack(padx=16, pady=5)
    status = tk.Label(root, text='Your token is entered locally and is not sent to the chat.', wraplength=540)
    status.pack(padx=16, pady=8)
    result = [None]

    def paste():
        try:
            value = root.clipboard_get()
        except tk.TclError:
            status.config(text='Clipboard has no text. Copy the token from Hugging Face first.')
            return
        entry.delete(0, tk.END)
        entry.insert(0, clean_token(value))

    def submit(event=None):
        value = clean_token(entry.get())
        if not value.startswith('hf_') or any(character.isspace() for character in value):
            status.config(text='Invalid format. Copy the FULL token beginning with hf_, not its name, URL or device code.')
            entry.focus_set()
            return
        result[0] = value
        entry.delete(0, tk.END)
        root.destroy()

    buttons = tk.Frame(root)
    buttons.pack(pady=8)
    tk.Button(buttons, text='Paste token', command=paste, width=18).pack(side='left', padx=8)
    tk.Button(buttons, text='Login and download', command=submit, width=22).pack(side='left', padx=8)
    root.bind('<Return>', submit)
    entry.focus_set()
    root.mainloop()
    return result[0]


def main():
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_DEBUG'] = '0'
    from huggingface_hub import login

    print('Create a Read token at https://huggingface.co/settings/tokens')
    print('Paste it below. Characters are hidden. Do not send the token in chat.')
    try:
        token = prompt_token()
    except (EOFError, KeyboardInterrupt):
        print('\nLogin cancelled.')
        return 1
    if not token:
        print('Login cancelled.')
        return 1
    try:
        login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
    except Exception as error:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        print('Login failed. Check the token and network connection.' + (f' HTTP {status}.' if status else ''))
        return 1
    finally:
        token = None
    print('Login successful. Starting model download next.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
