from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2_finetuned_amazon"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

available_categories = [
    "clothes", "electronics", "cars", "food", "toys", "furniture", "sport", "holidays", "kitchen", "health", "gaming", "jewelry", "writing", "makeup", "construction", "automotive", "fishing", "cleaning", "garden"
]

def generate_description_and_ad(product_name, category, max_length=128):
    input_text = f"Category: {category} | Product: {product_name} | Description:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")


    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "| Ad:" in generated_text:
        description = generated_text.split("| Description:")[1].split("| Ad:")[0].strip()
        ad = generated_text.split("| Ad:")[1].strip()
    else:
        description = generated_text.split("| Description:")[1].strip()
        ad = "Brak wygenerowanego hasła reklamowego."

    return description, ad

print("Dostępne kategorie:")
for i, category in enumerate(available_categories, 1):
    print(f"{i}. {category}")

category_choice = int(input("Wybierz numer kategorii: ")) - 1
selected_category = available_categories[category_choice]

product_name = input("Podaj nazwę produktu: ")

description, ad = generate_description_and_ad(product_name, selected_category)

print(f"\nProduct: {product_name}")
print(f"Category: {selected_category}")
print(f"Description: {description}")
print(f"Ad: {ad}")