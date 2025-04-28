import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import tqdm
import json
import time
import argparse  # Added for command-line arguments

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_reference_captions(descriptions_file):
    """
    Load reference captions from descriptions.txt file

    Expected format: image_id#caption_number caption_text
    Returns: Dictionary mapping image IDs to lists of captions
    """
    reference_captions = {}

    with open(descriptions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by first space to separate ID from caption
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue

            # Parse ID (format: image_id#caption_number)
            id_part = parts[0]
            caption = parts[1]

            # Extract image_id without caption number
            if '#' in id_part:
                img_id = id_part.split('#')[0]
            else:
                img_id = id_part

            # Add to dictionary
            if img_id not in reference_captions:
                reference_captions[img_id] = []
            reference_captions[img_id].append(caption)

    print(f"Loaded {len(reference_captions)} images with {sum(len(caps) for caps in reference_captions.values())} reference captions")
    return reference_captions

def calculate_bleu_scores(generated_captions_csv, reference_captions_dict, output_file="bleu_scores.csv"):
    """
    Calculate BLEU scores for generated captions against reference captions

    Args:
        generated_captions_csv: CSV file with generated captions (columns: image, caption)
        reference_captions_dict: Dictionary mapping image IDs to lists of reference captions
        output_file: Path to save results CSV

    Returns:
        DataFrame with results
    """
    # Load generated captions
    df_gen = pd.read_csv(generated_captions_csv)
    print(f"Loaded {len(df_gen)} generated captions")

    # Initialize results
    results = []
    smoother = SmoothingFunction().method1

    # Process each generated caption
    for _, row in tqdm.tqdm(df_gen.iterrows(), total=len(df_gen)):
        img_id = row['image']
        generated_caption = row['caption']

        # Extract image ID without extension if needed
        if '.' in img_id:
            img_id = os.path.splitext(img_id)[0]

        # Get reference captions
        references = reference_captions_dict.get(img_id, [])

        # Calculate BLEU scores
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0

        if references:
            # Tokenize
            candidate = nltk.word_tokenize(generated_caption.lower())
            references_tokenized = [nltk.word_tokenize(ref.lower()) for ref in references]

            # Calculate BLEU scores
            bleu1 = sentence_bleu(references_tokenized, candidate,
                                 weights=(1, 0, 0, 0), smoothing_function=smoother)
            bleu2 = sentence_bleu(references_tokenized, candidate,
                                 weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
            bleu3 = sentence_bleu(references_tokenized, candidate,
                                 weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
            bleu4 = sentence_bleu(references_tokenized, candidate,
                                 weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

        # Store results
        results.append({
            'image_id': img_id,
            'generated_caption': generated_caption,
            'reference_captions': references,
            'num_references': len(references),
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4
        })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Calculate average scores
    avg_scores = {
        'avg_bleu1': df_results['bleu1'].mean(),
        'avg_bleu2': df_results['bleu2'].mean(),
        'avg_bleu3': df_results['bleu3'].mean(),
        'avg_bleu4': df_results['bleu4'].mean(),
    }

    print("\nAverage BLEU Scores:")
    for metric, value in avg_scores.items():
        print(f"{metric}: {value:.4f}")

    # Save results
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save summary
    summary_file = output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'num_images': len(df_results),
            'avg_scores': avg_scores,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return df_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate BLEU scores for generated captions')

    parser.add_argument('--input', '-i',
                        required=True,
                        help='Path to CSV file with generated captions (columns: image, caption)')

    parser.add_argument('--references', '-r',
                        required=True,
                        help='Path to reference captions file (descriptions.txt)')

    parser.add_argument('--output', '-o',
                        default='bleu_evaluation.csv',
                        help='Path to save results CSV (default: bleu_evaluation.csv)')

    parser.add_argument('--show-top', '-t',
                        type=int,
                        default=5,
                        help='Number of top results to display (default: 5)')

    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # File paths from arguments
    generated_captions_csv = args.input
    descriptions_file = args.references
    output_file = args.output

    print(f"Input file: {generated_captions_csv}")
    print(f"References file: {descriptions_file}")
    print(f"Output file: {output_file}")

    # Load reference captions
    reference_captions = load_reference_captions(descriptions_file)

    # Calculate BLEU scores
    results_df = calculate_bleu_scores(
        generated_captions_csv=generated_captions_csv,
        reference_captions_dict=reference_captions,
        output_file=output_file
    )

    # Display top N results
    print(f"\nTop {args.show_top} results by BLEU-4 score:")
    top_results = results_df.sort_values('bleu4', ascending=False).head(args.show_top)
    for _, row in top_results.iterrows():
        print(f"Image: {row['image_id']}")
        print(f"Generated: {row['generated_caption']}")
        print(f"BLEU-4: {row['bleu4']:.4f}")
        print("References:")
        for ref in row['reference_captions'][:2]:  # Show only first 2 references
            print(f"  - {ref}")
        print()