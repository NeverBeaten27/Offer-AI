import os
import glob
from docx import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
import torch

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_and_prepare_data(docx_folder):
    files = glob.glob(os.path.join(docx_folder, '*.docx'))
    data = []
    for file in files:
        text = extract_text_from_docx(file)
        if text.strip():
            data.append({"text": text})
    return Dataset.from_list(data)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def train_ai(data, num_epochs=3):
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

def log_to_file(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(text + '\n')

def generate_offer(customer_requirements, template, log_file='generation_log.txt'):
    model_name = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    prompt = f"{template}\nCustomer Requirements: {customer_requirements}\nOffer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    attention_mask = (inputs.input_ids != tokenizer.pad_token_id).long()
    generated_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

    max_iterations = 20
    iteration = 0

    log_to_file(log_file, "Starting text generation...\n")

    while True:
        outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text += new_text[len(generated_text):]

        log_to_file(log_file, f"Iteration {iteration}: Generated {len(new_text.split())} new words.")
        log_to_file(log_file, f"Generated text so far: {generated_text}\n")

        if new_text.strip().endswith('.') or len(tokenizer(generated_text, return_tensors="pt").input_ids[0]) >= 1024:
            break

        iteration += 1
        if iteration >= max_iterations:
            log_to_file(log_file, "Reached maximum iterations.")
            break

        inputs = tokenizer(generated_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        attention_mask = (inputs.input_ids != tokenizer.pad_token_id).long()

    return generated_text[len(prompt):].strip()

def display_menu():
    print("######################### Welcome to Offer AI #########################\n\n################################ Menu #################################")
    print("1. Extract data")
    print("2. Train AI")
    print("3. Generate offer")
    print("4. Help")
    print("5. Exit\n")

def get_menu_choice():
    while True:
        try:
            menu = int(input("Please input menu number: "))
            if menu < 1 or menu > 5:
                print("Invalid choice. Please choose a number between 1 and 5.")
            else:
                return menu
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    display_menu()
    menu = 0
    template = ""
    while menu != 5:
        menu = get_menu_choice()
        if menu == 1:
            docx_folder = input("Please input the folder containing the .docx files: ")
            data = load_and_prepare_data(docx_folder)
            print("Data extracted successfully.")
        elif menu == 2:
            num_epochs = int(input("Please input the number of training epochs: "))
            train_ai(data, num_epochs)
            template = data["text"][0] if data else ""
            print("Template created successfully.")
        elif menu == 3:
            customer_requirements = input("Please input the customer's requirements: ")
            if template:
                offer = generate_offer(customer_requirements, template)
                print("Generated Offer:\n", offer)
            else:
                print("Please create a template first by choosing option 2.")
        elif menu == 4:
            display_menu()
        elif menu == 5:
            break
        else:
            print("Please input a number between 1 and 5")
