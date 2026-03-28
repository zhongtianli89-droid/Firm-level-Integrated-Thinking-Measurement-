# =========================================================
# Step 1: Import necessary libraries and lists
# =========================================================

import os
import re
from collections import Counter
import pandas as pd
from nltk.tokenize import RegexpTokenizer
# =========================================================
# Step 2: Extract corporate participants from file content
# =========================================================
def extract_corporate_participants(file_content):
    participants_section = re.search(r'Corporate Participants\s*=+\n(.*?)\n={3,}', file_content, re.S)
    if not participants_section:
        return []
    return re.findall(r'\* (.*?)\n', participants_section.group(1))

# =========================================================
# Step 3: Extract Q&A section from file content
# =========================================================
def extract_qa_section(file_content):
    qa_match = re.search(r'Questions and Answers\s*\n-+\n(.*)', file_content, re.S)
    return qa_match.group(1) if qa_match else ""

# =========================================================
# Step 4: Extract Q&A sentences based on participants
# =========================================================
def extract_qa_for_participants(qa_content, corporate_names):
    qa_blocks = re.split(r'\n-+\n', qa_content)
    relevant_data = {name: [] for name in corporate_names}
    for i in range(len(qa_blocks) - 1):
        block = qa_blocks[i].strip()
        next_block = qa_blocks[i + 1].strip()
        lines = block.splitlines()
        if lines:
            speaker_match = re.match(r'^(.*?),', lines[0].strip())
            if speaker_match and speaker_match.group(1).strip() in corporate_names:
                relevant_data[speaker_match.group(1).strip()].append(next_block)
    return relevant_data

# =========================================================
# Step 5: Extract presentation section from file content
# =========================================================
def extract_presentation_section(file_content):
    has_qa_section = re.search(r'Questions and Answers\s*\n-+\n', file_content, re.S)

    if has_qa_section:
        presentation_match = re.search(r'Presentation\s*\n-+\n(.*?)(?=\n={3,})', file_content, re.S)
        return presentation_match.group(1).strip() if presentation_match else ""
    else:
        fallback_match = re.search(r'Presentation\s*\n-+\n(.*)', file_content, re.S)
        return fallback_match.group(1).strip() if fallback_match else ""

# =========================================================
# Step 6: Extract Presentation sentences based on participants
# =========================================================
def extract_presentation_for_participants(presentation_content, corporate_names):
    presentation_blocks = re.split(r'\n-+\n', presentation_content)
    relevant_data = {name: [] for name in corporate_names}
    for i in range(len(presentation_blocks) - 1):
        block = presentation_blocks[i].strip()
        next_block = presentation_blocks[i + 1].strip()
        lines = block.splitlines()
        if lines:
            speaker_match = re.match(r'^(.*?),', lines[0].strip())
            if speaker_match and speaker_match.group(1).strip() in corporate_names:
                relevant_data[speaker_match.group(1).strip()].append(next_block)
    return relevant_data

# =========================================================
# Step 7: Tokenizing and calculating word counts
# =========================================================
def clean_text(content, stop_words):
    content = content.lower()
    content = re.sub(r'[^a-z\s\-]', ' ', content)
    tokenizer = RegexpTokenizer(r'\b\w[\w-]*\b')
    tokens = tokenizer.tokenize(content)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def tokenize_and_count(data, stop_words):
    tokenized_data = {}
    total_word_count = {}
    for participant, text_blocks in data.items():
        cleaned_text_value = ' '.join([clean_text(block, stop_words) for block in text_blocks])
        tokenized_data[participant] = cleaned_text_value
        total_word_count[participant] = len(cleaned_text_value.split())
    return tokenized_data, total_word_count

# =========================================================
# Step 8: Count lexicon near multiple context words in one run
# =========================================================
def count_lexicon_near_context_words(text, lexicon, context_words, window_length=10):
    tokens = re.findall(r'\b\w[\w\-]*\b', text.lower())
    windows = {"positive": [], "negative": []}
    context_categories = {
        "positive": context_words["positive"],
        "negative": context_words["negative"]
    }

    # Extract sliding windows for each category
    for i, word in enumerate(tokens):
        for category, words in context_categories.items():
            if word in words:
                start = max(0, i - window_length)
                end = min(len(tokens), i + window_length + 1)
                windows[category].append(tokens[start:end])

    # Count lexicon occurrences for each category
    lexicon_counts = {category: Counter() for category in windows}
    sorted_lexicon = sorted(lexicon, key=len, reverse=True)

    for category, category_windows in windows.items():
        for window in category_windows:
            window_text = " ".join(window)
            for term in sorted_lexicon:
                term_pattern = re.escape(term)
                term_regex = re.compile(rf'(?<!\w){term_pattern}(?![\w\-])', re.IGNORECASE)
                matches = term_regex.findall(window_text)
                lexicon_counts[category][term] += len(matches)

    return lexicon_counts

# =========================================================
# Step 9: Count lexicon without context
# =========================================================
def count_lexicon_occurrences(text, lexicon):
    count = Counter()
    sorted_lexicon = sorted(lexicon, key=len, reverse=True)
    for term in sorted_lexicon:
        term_pattern = re.escape(term)
        term_regex = re.compile(rf'(?<!\w){term_pattern}(?![\w\-])', re.IGNORECASE)
        matches = term_regex.findall(text)
        count[term] += len(matches)
    return count

# =========================================================
# Step 10: Fully integrated metadata parsing
# =========================================================
def parse_file_name(file_name):
    components = file_name.split('-')
    return {
        "File Name": file_name,
        "Year": components[0],
        "Quarter": components[1],
        "Company Code": components[3],
        "Numeric Code": components[4].split('.')[0]
    }

# =========================================================
# Step 11: Loading the required TXT files
# =========================================================
def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip().lower() for line in file if line.strip())

def load_all_context_files(stop_words_file, pos_tone_file, neg_tone_file, lexicon_file):
    stop_words = load_txt_file(stop_words_file)
    positive_words = load_txt_file(pos_tone_file)
    negative_words = load_txt_file(neg_tone_file)
    lexicon = load_txt_file(lexicon_file)

    context_words = {
        "positive": positive_words,
        "negative": negative_words
    }

    return stop_words, context_words, lexicon

stop_words, context_words, lexicon = load_all_context_files(
    stop_words_file="",
    pos_tone_file="",
    neg_tone_file="",
    lexicon_file="")

transcript_folder = "/Users/zhongtian_li/Desktop/2017 scripts"

# =========================================================
# Helper 1: keep only identified terms with nonzero counts
# =========================================================
def filter_nonzero_counts(counter_obj):
    return {term: count for term, count in counter_obj.items() if count > 0}

# =========================================================
# Helper 2: Layer 1 analytic identification for a single file
# =========================================================
def layer1_identification_for_file(file_path, lexicon, context_words, stop_words):
    file_name = os.path.basename(file_path)
    metadata = parse_file_name(file_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Extract structural sections
    participants = extract_corporate_participants_with_fallback(file_path, file_content)
    qa_section = extract_qa_section(file_content)
    presentation_section = extract_presentation_section(file_content)

    # Clean full section text (same logic as your original code)
    qa_cleaned_text = clean_text(qa_section, stop_words)
    presentation_cleaned_text = clean_text(presentation_section, stop_words)

    qa_word_count = len(qa_cleaned_text.split())
    presentation_word_count = len(presentation_cleaned_text.split())

    # Context-based identification
    qa_context_counts = count_lexicon_near_context_words(qa_cleaned_text, lexicon, context_words)
    presentation_context_counts = count_lexicon_near_context_words(presentation_cleaned_text, lexicon, context_words)

    # Proportions (same logic as original code)
    qa_proportions = {
        category: sum(counts.values()) / qa_word_count if qa_word_count > 0 else 'Missing'
        for category, counts in qa_context_counts.items()
    }
    presentation_proportions = {
        category: sum(counts.values()) / presentation_word_count if presentation_word_count > 0 else 'Missing'
        for category, counts in presentation_context_counts.items()
    }

    # No-context identification
    qa_lexicon_counts = count_lexicon_occurrences(qa_cleaned_text, lexicon)
    presentation_lexicon_counts = count_lexicon_occurrences(presentation_cleaned_text, lexicon)

    qa_lexicon_proportion = sum(qa_lexicon_counts.values()) / qa_word_count if qa_word_count > 0 else 'Missing'
    presentation_lexicon_proportion = sum(presentation_lexicon_counts.values()) / presentation_word_count if presentation_word_count > 0 else 'Missing'

    # Return both summary metrics and detailed identified terms
    result = {
        "metadata": {
            **metadata,
            "Corporate Participants": participants
        },
        "cleaned_text": {
            "qa_cleaned_text": qa_cleaned_text,
            "presentation_cleaned_text": presentation_cleaned_text
        },
        "summary": {
            "Q&A Total Word Count": qa_word_count,
            "Presentation Total Word Count": presentation_word_count,
            "QA Positive Proportion": qa_proportions["positive"],
            "QA Negative Proportion": qa_proportions["negative"],
            "Presentation Positive Proportion": presentation_proportions["positive"],
            "Presentation Negative Proportion": presentation_proportions["negative"],
            "QA Lexicon Proportion (No Context)": qa_lexicon_proportion,
            "Presentation Lexicon Proportion (No Context)": presentation_lexicon_proportion
        },
        "identified_terms": {
            "qa_context_terms": {
                "positive": filter_nonzero_counts(qa_context_counts["positive"]),
                "negative": filter_nonzero_counts(qa_context_counts["negative"])
            },
            "presentation_context_terms": {
                "positive": filter_nonzero_counts(presentation_context_counts["positive"]),
                "negative": filter_nonzero_counts(presentation_context_counts["negative"])
            },
            "qa_lexicon_terms_no_context": filter_nonzero_counts(qa_lexicon_counts),
            "presentation_lexicon_terms_no_context": filter_nonzero_counts(presentation_lexicon_counts)
        }}
    

    return result

# =========================================================
# Helper 3: Extract participants with fallback to brief file
# =========================================================
def extract_corporate_participants_with_fallback(file_path, file_content):
    participants = extract_corporate_participants(file_content)

    # Case 1: Transcript already has participants → use it
    if participants:
        return participants

    folder_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    # Replace transcript.txt with brief.txt
    brief_file_name = re.sub(r'transcript\.txt$', 'brief.txt', file_name, flags=re.IGNORECASE)
    brief_file_path = os.path.join(folder_path, brief_file_name)

    # Case 2: Try fallback to brief file
    if os.path.exists(brief_file_path):
        with open(brief_file_path, 'r', encoding='utf-8', errors='ignore') as brief_file:
            brief_content = brief_file.read()

        brief_participants = extract_corporate_participants(brief_content)

        if brief_participants:
            print(f"Fallback to brief file used for: {file_name}")
            return brief_participants
        else:
            print(f"Fallback attempted but no participants found in brief file: {file_name}")
    else: print(f"No brief file found for: {file_name}")

    # Case 3: No brief file or still no participants
    return []

# =========================================================
# Helper 4: Folder-level Layer 1 processing
# Saves both summary table and detailed identifications
# =========================================================
def process_and_save_layer1_results(folder_path, lexicon, context_words, stop_words):
    summary_results = []
    detailed_results = []

    total_files = len([f for f in os.listdir(folder_path) if f.lower().endswith('transcript.txt')])
    processed_files = 0

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('transcript.txt'):
            processed_files += 1
            print(f"Processing file {processed_files}/{total_files}: {file_name}")

            file_path = os.path.join(folder_path, file_name)
            result = layer1_identification_for_file(file_path, lexicon, context_words, stop_words)

            # Summary row
            summary_row = {
                **result["metadata"],
                **result["summary"]
            }
            summary_results.append(summary_row)

            # Detailed row
            detailed_row = {
                "File Name": result["metadata"]["File Name"],
                "Year": result["metadata"]["Year"],
                "Quarter": result["metadata"]["Quarter"],
                "Company Code": result["metadata"]["Company Code"],
                "Numeric Code": result["metadata"]["Numeric Code"],
                "qa_context_terms": result["identified_terms"]["qa_context_terms"],
                "presentation_context_terms": result["identified_terms"]["presentation_context_terms"],
                "qa_lexicon_terms_no_context": result["identified_terms"]["qa_lexicon_terms_no_context"],
                "presentation_lexicon_terms_no_context": result["identified_terms"]["presentation_lexicon_terms_no_context"]
            }
            detailed_results.append(detailed_row)

    summary_df = pd.DataFrame(summary_results)
    detailed_df = pd.DataFrame(detailed_results)

    output_base = os.path.basename(folder_path.rstrip(os.sep))

    summary_csv = f"{output_base}_layer1_summary.csv"
    summary_pkl = f"{output_base}_layer1_summary.pkl"
    detailed_csv = f"{output_base}_layer1_detailed.csv"
    detailed_pkl = f"{output_base}_layer1_detailed.pkl"

    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_pickle(summary_pkl)

    detailed_df.to_csv(detailed_csv, index=False)
    detailed_df.to_pickle(detailed_pkl)

    print(f"Summary results saved as CSV: {summary_csv}")
    print(f"Summary results saved as Pickle: {summary_pkl}")
    print(f"Detailed identification results saved as CSV: {detailed_csv}")
    print(f"Detailed identification results saved as Pickle: {detailed_pkl}")
    
if __name__ == "__main__":
    process_and_save_layer1_results(
        folder_path=transcript_folder,
        lexicon=lexicon,
        context_words=context_words,
        stop_words=stop_words
    )
    
    
    