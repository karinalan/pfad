import random

# Define lists of Italian words starting with 'e', 'g', 'i', 'l', 'o', 'v'
word_list = {
    'e': ['elefante', 'energia', 'estate', 'esempio', 'errore'],
    'g': ['gelato', 'gatto', 'giorno', 'giardino', 'giraffa'],
    'i': ['idea', 'isola', 'insegnante', 'inchiostro', 'insetto'],
    'l': ['lamponi', 'luna', 'limone', 'libro', 'leone'],
    'o': ['occhio', 'ombra', 'orto', 'onda', 'orologio'],
    'v': ['vino', 'vela', 'vento', 'valle', 'vittoria']
}

# Track words collected by the user
collected_words = {
    'e': None,
    'g': None,
    'i': None,
    'l': None,
    'o': None,
    'v': None
}

# Function to check if the user has collected all words
def check_achievement():
    return all(collected_words.values())  # Returns True if all letters have a word

print("Welcome to the word guessing game! Guess a word from the 26 letters, but only 'e', 'g', 'i', 'l', 'o', 'v' will give you a word.")

# Game loop
while True:
    letter = input("Enter a letter: ").lower()

    if letter in word_list:
        # Check if a word for this letter has already been collected
        if collected_words[letter]:
            print(f"You have already collected the word for the letter '{letter}': {collected_words[letter]}")
        else:
            # Randomly select a word and store it
            word = random.choice(word_list[letter])
            collected_words[letter] = word
            print(f"You collected the word for the letter '{letter}': {word}")
        
        # Check if the achievement is unlocked
        if check_achievement():
            print("\nCongratulations! You have unlocked the 'LOVE GIO 🩵' achievement!")
            break
    else:
        # Feedback for invalid letter input
        print("haha! NOOOO!!!")
