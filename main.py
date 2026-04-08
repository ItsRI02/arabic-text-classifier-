import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import time
import pandas as pd
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from arabert.preprocess import ArabertPreprocessor
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'text_classification'
}
STOP_WORDS_TABLE = 'stopwords'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TextClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Text Classification")
        self.model_type = tk.StringVar(value="fine_tuned")
        self.corpus_path = None
        self.arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)
        self.stop_words = self.get_stopwords()
        self.label_encoder = LabelEncoder()
        self.train_texts, self.train_labels, self.test_data = [], [], []
        self.prediction_results_df = None

        self.progress_label = None
        self.progress = None

        self.create_start_screen()

    def set_background(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((1200, 800))  # Match your root.geometry
            bg_image = ImageTk.PhotoImage(image)
            self.bg_image = bg_image  # prevent garbage collection

            bg_label = tk.Label(self.root, image=bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            return bg_label
        except Exception as e:
            print(f"Error loading background image: {e}")

    def confirm_exit(self):
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            self.root.destroy()

    def start_classification_with_progress(self):
        if not self.corpus_path:
            messagebox.showerror("Error", "Please select a corpus folder.")
            return

        self.progress_label.pack(pady=5)
        self.progress.pack(pady=5)
        self.progress.start()
        threading.Thread(target=self.run_classification).start()

    def get_stopwords(self):
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT word FROM {STOP_WORDS_TABLE}")
        stop_words = {row[0] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return stop_words

    def create_start_screen(self):
        self.clear_screen()
        self.set_background("bk-app (3).png")
        frame = tk.Frame(self.root, bg="")
        frame.place(relx=0.5, rely=0.6, anchor='center')

        tk.Label(frame, text="Select Model Type", font=("Arial", 18), bg="white").pack(pady=10)
        tk.Radiobutton(frame, text="Fine-tuned AraBERT", variable=self.model_type, value="fine_tuned",
                       font=("Arial", 14), bg="white").pack(anchor='center')
        tk.Radiobutton(frame, text="Non Fine-tuned AraBERT", variable=self.model_type, value="non_fine_tuned",
                       font=("Arial", 14), bg="white").pack(anchor='center')

        tk.Button(frame, text="Next", command=self.create_select_corpus_screen, font=("Arial", 14),
                  bg="#FFDB58", fg="black", width=20, height=2).pack(pady=40)
        tk.Button(frame, text="Exit", command=self.confirm_exit, font=("Arial", 12),
                  bg="red", fg="white", width=10, height=1).pack()

    def create_select_corpus_screen(self):
        self.clear_screen()
        self.set_background("bk-app (3).png")
        frame = tk.Frame(self.root)
        frame.place(relx=0.5, rely=0.6, anchor='center')

        tk.Label(frame, text="Select Corpus Folder", font=("Arial", 16)).pack(pady=20)
        tk.Button(frame, text="Browse", command=self.browse_folder, font=("Arial", 14), bg="#FFDB58", fg="black").pack(pady=10)

        self.selected_path_lbl = tk.Label(frame, text="No folder selected", font=("Arial", 12), wraplength=400)
        self.selected_path_lbl.pack(pady=5)

        tk.Button(frame, text="Start Classification", command=self.start_classification_with_progress,
                  font=("Arial", 16), width=15, height=2, bg="#FFDB58", fg="black").pack(pady=10)

        self.progress_label = tk.Label(frame, text="Please wait...", font=("Arial", 12), fg="gray")
        self.progress = ttk.Progressbar(frame, mode='indeterminate', length=300)

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.corpus_path = path
            self.selected_path_lbl.config(text=path)

    def run_classification(self):
        if not self.corpus_path:
            messagebox.showerror("Error", "Please select a corpus folder.")
            return

        self.load_data(self.corpus_path)

        if not self.train_texts or not self.test_data:
            messagebox.showerror("Error", "No valid train/test data found in selected folder.")
            return

        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(self.train_labels)
        test_texts = [text for text, label, fname in self.test_data]
        test_labels = [label for _, label, _ in self.test_data]
        test_labels_encoded = self.label_encoder.transform(test_labels)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels):
                self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        train_dataset = TextDataset(self.train_texts, train_labels_encoded)
        test_dataset = TextDataset(test_texts, test_labels_encoded)

        # Load pretrained AraBERT model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.label_encoder.classes_)
        ).to(device)

        if self.model_type.get() == "fine_tuned":
            # Fine-tuning setup
            training_args = TrainingArguments(
                output_dir='./results_test',
                num_train_epochs=2,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                logging_dir='./logs_test',
                logging_steps=10,
                load_best_model_at_end=False,
            )

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
                acc = accuracy_score(labels, preds)
                return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics
            )
            trainer.train()
            metrics = trainer.evaluate()
            predictions = trainer.predict(test_dataset).predictions

        else:
            # Non fine-tuned mode: inference only
            model.eval()
            inputs = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=128).to(
                device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.time()
            runtime_seconds = end_time - start_time

            predictions = outputs.logits.cpu().numpy()
            preds = torch.argmax(torch.tensor(predictions), dim=1).numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels_encoded, preds, average='weighted')
            acc = accuracy_score(test_labels_encoded, preds)
            metrics = {
                'eval_accuracy': acc,
                'eval_precision': precision,
                'eval_recall': recall,
                'eval_f1': f1
            }
            metrics['runtime'] = runtime_seconds / 60

        # Map predicted class indices to class labels
        predicted_labels = torch.argmax(torch.tensor(predictions), dim=1).numpy()
        predicted_classes = self.label_encoder.inverse_transform(predicted_labels)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_type = self.model_type.get()

        # Save results to MySQL database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Save evaluation results
        cursor.execute("""
                    INSERT INTO evaluation_results (model_type, accuracy, precision_score, recall, f1_score, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
            model_type, metrics['eval_accuracy'], metrics['eval_precision'],
            metrics['eval_recall'], metrics['eval_f1'], timestamp
        ))

        # Save individual predictions
        for i, (text, true_label, fname) in enumerate(self.test_data):
            cursor.execute("""
                        INSERT INTO prediction_results (file_name, true_label, predicted_label, model_type, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (fname, true_label, predicted_classes[i], model_type, timestamp))

        # Compute confusion matrix and align labels
        labels = self.label_encoder.classes_  # Ensures consistent label order
        cm = confusion_matrix(test_labels, predicted_classes, labels=labels)

        # Save confusion matrix to DB
        for i in range(len(labels)):
            for j in range(len(labels)):
                count = cm[i][j]
                if count > 0:
                    cursor.execute("""
                                INSERT INTO confusion_matrix (true_label, predicted_label, count, model_type, timestamp)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (labels[i], labels[j], int(count), model_type, timestamp))

        conn.commit()
        cursor.close()
        conn.close()

        # Display results in the GUI
        self.prediction_results_df = pd.DataFrame({
            "File": [fname for _, _, fname in self.test_data],
            "True Label": test_labels,
            "Predicted Label": predicted_classes
        })

        self.root.after(0, self.show_results_screen, metrics, cm, labels)

    def load_data(self, folder_path):
        self.train_texts.clear()
        self.train_labels.clear()
        self.test_data.clear()
        for fname in os.listdir(folder_path):
            if fname.endswith('.txt'):
                parts = fname.split('_')
                if len(parts) < 3:
                    continue  # skip malformed filenames
                prefix = parts[0]
                label = parts[2].lower()
                file_path = os.path.join(folder_path, fname)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    # Remove stopwords by tokenizing and filtering
                    tokens = text.split()
                    tokens = [t for t in tokens if t not in self.stop_words]
                    cleaned = self.arabert_prep.preprocess(" ".join(tokens))
                    if prefix == 'L':
                        self.train_texts.append(cleaned)

                        self.train_labels.append(label.lower())
                    elif prefix == 'T':
                        self.test_data.append((cleaned, label.lower(), fname))

    def create_scrollable_frame(self):
        canvas = tk.Canvas(self.root)
        scrollbar_y = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollbar_x = tk.Scrollbar(self.root, orient="horizontal", command=canvas.xview)

        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        return scroll_frame

    def show_results_screen(self, metrics, cm, labels):
        self.clear_screen()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Left column: Evaluation metrics + Confusion matrix
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill='y', padx=20)

        tk.Label(left_frame, text="Evaluation Metrics", font=("Arial", 18, "bold")).pack(pady=10, anchor='w')
        for key, value in metrics.items():
            key_lower = key.lower()
            if key_lower in ["accuracy", "precision", "recall", "f1", "eval_accuracy", "eval_loss", "eval_precision",
                             "eval_recall", "eval_f1"]:
                percentage = value * 100
                tk.Label(left_frame, text=f"{key}: {percentage:.2f}%", font=("Arial", 14)).pack(anchor='w')
            elif "runtime" in key_lower:
                tk.Label(left_frame, text=f"{key}: {value:.2f} min", font=("Arial", 14)).pack(anchor='w')
            else:
                tk.Label(left_frame, text=f"{key}: {value}", font=("Arial", 14)).pack(anchor='w')

        tk.Label(left_frame, text="Confusion Matrix", font=("Arial", 18, "bold")).pack(pady=20, anchor='w')

        fig, ax = plt.subplots(figsize=(5, 4))  # Smaller size to fit screen
        cax = ax.matshow(cm, cmap='YlOrBr')
        fig.colorbar(cax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        normed_cm = cax.norm
        cmap = cax.cmap

        for i in range(len(labels)):
            for j in range(len(labels)):
                cell_value = cm[i][j]
                color = cmap(normed_cm(cell_value))  # ✅ FIXED: Call norm as function
                r, g, b, _ = color
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'black' if brightness > 0.5 else 'white'
                ax.text(j, i, str(cell_value), va='center', ha='center', color=text_color, fontsize=10)

        chart = FigureCanvasTkAgg(fig, master=left_frame)
        chart.draw()
        chart.get_tk_widget().pack(pady=10)

        # Right column: Prediction results table
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill='y', padx=20)

        tk.Label(right_frame, text="Prediction Results", font=("Arial", 18, "bold")).pack(pady=10)

        table_frame = tk.Frame(right_frame)
        table_frame.pack()

        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 14))  # Increase row font
        style.configure("Treeview.Heading", font=("Arial", 14, "bold"))  # Optional: increase header font

        columns = ("File", "True Label", "Predicted Label")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=30)  # Increase height
        tree.tag_configure('correct', background='#d4edda')
        tree.tag_configure('wrong', background='#f8d7da')

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for _, row in self.prediction_results_df.iterrows():
            true_label = row["True Label"]
            predicted_label = row["Predicted Label"]
            tag = 'correct' if true_label == predicted_label else 'wrong'
            tree.insert("", tk.END, values=(row["File"], true_label, predicted_label), tags=(tag,))

        tree.pack(side=tk.LEFT)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = tk.Frame(main_frame)
        button_frame.place(relx=0.5, rely=0.97, anchor='s')

        tk.Button(button_frame, text="Back to Start", command=self.create_start_screen,
                  font=("Arial", 14), bg="#FFDB58", fg="black").pack()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = TextClassifierApp(root)
    root.geometry("1200x760")
    root.mainloop()

