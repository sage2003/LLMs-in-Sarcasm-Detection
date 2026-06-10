import torch
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from datasets import load_dataset
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SarcasmBinaryClassifier:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the sarcasm binary classification framework.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Data storage
        self.sarcasm_data = []
        self.predictions = []
        self.classification_results = {}
        
    def _load_model(self):
        """Load the model and tokenizer.
        Try to load from Hugging Face first, fall back to local copy if it fails.
        """
        logger.info(f"Attempting to load model: {self.model_name}")

        local_model_path = "llama/"  # Fallback path
        hf_load_failed = False

        try:
            # --- Try loading from Hugging Face ---
            logger.info(f"Trying to load model from Hugging Face: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully from Hugging Face.")

        except Exception as e:
            hf_load_failed = True
            logger.warning(f"Failed to load model from Hugging Face: {e}")
            logger.info("Falling back to local model...")

        if hf_load_failed:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                logger.info("Local model loaded successfully.")
            except Exception as e_local:
                logger.error(f"Failed to load both Hugging Face and local model: {e_local}")
                raise RuntimeError("Model loading failed from both sources.") from e_local

        # Move to device if needed
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model ready for inference.")


    
    def load_sarcasm_datasets(self):
        """
        Load sarcasm dataset with fallback options.
        """
        logger.info("Loading sarcasm dataset...")
        dataset = None

        
        try:
            local_path = "Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv"
            # local_path = "Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv"
            df = pd.read_csv(local_path)
            logger.info(f"  === Loaded local CSV with {len(df)} samples ===")
            dataset = df.to_dict(orient="records")
        except Exception as e2:
            logger.warning(f"  Local CSV also not found: {e2}")
            return

        if dataset is not None:
            if isinstance(dataset, list):
                records = dataset
            else:
                records = [dict(item) for item in dataset]

            self.sarcasm_data = [
                {
                    #"context": item.get("context", "").strip("[]'\""),
                    "sentence": item.get("text", "").strip("[]'\""),
                    "true_label": 1 if str(item.get("class", "")).strip().lower() == "sarcastic" else 0,
                    "source": "IAC-V2"
                }
                for item in records
                if isinstance(item.get("text", ""), str) and 5 <= len(item["text"].split()) <= 100
            ]

            logger.info(f"Processed {len(self.sarcasm_data)} usable examples.")

    
    def classify_sarcasm(self, num_examples: int = 100):
        """
        Perform binary classification on sarcasm detection.
        """
        logger.info("Starting binary classification...")
        
        if len(self.sarcasm_data) == 0:
            logger.error("No data available for classification")
            return
        
        # Limit to available examples
        num_examples = min(num_examples, len(self.sarcasm_data))
        test_data = self.sarcasm_data[:num_examples]
        
        results = []
        for example in tqdm(test_data, desc="Classifying"):
            result = self._classify_single_example(example)
            results.append(result)
        
        self.predictions = results
        self._analyze_classification_results()
        self._plot_classification_results()
        
        return self.classification_results
    
    def _classify_single_example(self, example: Dict) -> Dict:
        """
        Classify a single example using binary classification.
        """
        #context = example['context']
        sentence = example['sentence']
        true_label = example['true_label']
        
        # # 1. Define the prompt as a chat structure
        # messages = [
        #     {
        #         "role": "system",
        #         # Enhanced system prompt: Defines persona, exact task, and strict output constraints (positive and negative).
        #         "content": "You are an expert sarcasm classifier. Your sole task is to analyze the input sentence and output a single word: 'Yes' if it is sarcastic, or 'No' if it is not. Do not provide standard prose, explanations, or punctuation beyond the single word."
        #     },
        #     {
        #         "role": "user",
        #         # Enhanced user prompt: Separates instruction from data and reinforces the constraint at the very end.
        #         "content": f"""Analyze the following sentence for sarcasm:

        # Sentence: "{sentence}"

        # Remember, output ONLY 'Yes' or 'No'."""
        #     }
        # ]

        # # 2. Apply the model's specific chat template
        # prompt = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )

        # prompt += "Answer:"
        #  For Qwen2.5-14B-Instruct
        # prompt = f"""You are known for being able to precisely classify whether a sentence is sarcastic or not. Determine whether the sentence is sarcastic.

        #      Sentence: "{sentence}"

        #      Is the sentence sarcastic? Answer strictly with only "Yes" or "No": """

# For Llama-3.1-8B-Instruct
        prompt = f"""<|start_header_id|>system<|end_header_id|>           

Your task is to classify if a sentence is sarcastic. Analyze the given sentence. Sarcasm often involves saying the opposite of what is true, using over-exaggeration, or conveying mockery. If the intent is to mock or criticize through irony, it is sarcastic.

Answer with only "Yes" or "No".

Sentence: {sentence}<|eot_id|><|start_header_id|>user<|end_header_id|>

Is the response sarcastic?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# For Phi-3-medium-4k-instructs

#         prompt = f"""Is the following sentence sarcastic? Answer Yes or No Only, no other text.

# Sentence: "{sentence}"

# Answer:"""

# For Gemma-2-9b-it

#         prompt = f"""Answer Yes or No Only, no other text.

# Rules:
# - Determine whether the sentence is sarcastic or not.
# - Answer with exactly one word: Yes or No.
# - If you are unsure, answer No.

# Sentence: "{sentence}"

# Answer:"""


        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,     
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                
                gen_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
                answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()


                
                # Map answer to prediction
                if 'yes' in answer:
                    predicted_label = 1
                    confidence = 1.0
                elif 'no' in answer:
                    predicted_label = 0
                    confidence = 1.0
                else:
                    # If model doesn't follow instructions, use fallback
                    predicted_label = 0
                    confidence = 0.0
                    logger.warning(f"Model didn't follow instructions. Output: '{answer}'")
                
                is_correct = predicted_label == true_label
                
                return {
                    'sentence': sentence,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'model_output': answer,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'prompt': prompt
                }
                
        except Exception as e:
            logger.error(f"Error classifying example: {e}")
            return {
                'sentence': sentence,
                'true_label': true_label,
                'predicted_label': -1,  # Error indicator
                'model_output': 'ERROR',
                'confidence': 0.0,
                'is_correct': False,
                'error': str(e)
            }
    
    def _analyze_classification_results(self):
        """
        Analyze classification results and compute metrics.
        """
        # Filter out errors
        valid_results = [r for r in self.predictions if r['predicted_label'] != -1]
        
        if not valid_results:
            logger.error("No valid results to analyze")
            return
        
        y_true = [r['true_label'] for r in valid_results]
        y_pred = [r['predicted_label'] for r in valid_results]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        self.classification_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_examples': len(valid_results),
            'correct_predictions': sum([r['is_correct'] for r in valid_results]),
            'detailed_results': valid_results
        }
        
        logger.info("Classification Analysis Complete:")
        logger.info(f"  - Accuracy: {accuracy:.3f}")
        logger.info(f"  - Precision: {precision:.3f}")
        logger.info(f"  - Recall: {recall:.3f}")
        logger.info(f"  - F1 Score: {f1:.3f}")
        logger.info(f"  - Correct: {self.classification_results['correct_predictions']}/{len(valid_results)}")
    
    def _plot_classification_results(self):
        """
        Create classification performance visualizations.
        """
        if not self.classification_results:
            return
        
        cm = np.array(self.classification_results['confusion_matrix'])
        metrics = self.classification_results
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Not Sarcastic', 'Sarcastic'],
                   yticklabels=['Not Sarcastic', 'Sarcastic'])
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_title('Confusion Matrix')
        
        # Plot 2: Metrics Comparison
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Classification Metrics')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Correct vs Incorrect Predictions
        correct = metrics['correct_predictions']
        incorrect = metrics['total_examples'] - correct
        
        axes[1, 0].pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Prediction Accuracy')
        
        # Plot 4: Class-wise Performance
        report = metrics['classification_report']
        class_0_metrics = [report['0']['precision'], report['0']['recall'], report['0']['f1-score']]
        class_1_metrics = [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]
        
    
        x = np.arange(3)
        width = 0.35
        
        axes[1, 1].bar(x - width/2, class_0_metrics, width, label='Not Sarcastic', alpha=0.8)
        axes[1, 1].bar(x + width/2, class_1_metrics, width, label='Sarcastic', alpha=0.8)
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Class-wise Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('binary_classification_results_kimi-k2-instruct-0905_Llama-3.1-8B-Instruct.png', dpi=500, bbox_inches='tight')
        plt.close()

        logger.info("Classification results plots saved as 'binary_classification_results_kimi-k2-instruct-0905_Llama-3.1-8B-Instruct.png'")
    
    def save_results(self, filename: str = "binary_classification_results_kimi-k2-instruct-0905_Llama-3.1-8B-Instruct.json"):
        """
        Save classification results to JSON file.
        """
        results = {
            'model_name': self.model_name,
            'classification_metrics': {
                'accuracy': self.classification_results.get('accuracy', 0),
                'precision': self.classification_results.get('precision', 0),
                'recall': self.classification_results.get('recall', 0),
                'f1_score': self.classification_results.get('f1_score', 0),
            },
            'confusion_matrix': self.classification_results.get('confusion_matrix', []),
            'sample_predictions': self.predictions[:20],  # Save first 20 for inspection
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_tested': len(self.predictions)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save detailed predictions to CSV
        df = pd.DataFrame(self.predictions)
        csv_filename = filename.replace('.json', '_detailed.csv')
        df.to_csv(csv_filename, index=False)
        logger.info(f"Detailed predictions saved to {csv_filename}")

def main():
    """
    Main execution function.
    """
    logger.info("Starting Sarcasm Binary Classification")
    
    # Initialize classifier     #meta-llama/Llama-3.1-8B-Instruct       #Qwen/Qwen2.5-14B-Instruct      #microsoft/Phi-3-medium-4k-instruct     #google/gemma-2-9b-it
    classifier = SarcasmBinaryClassifier(model_name="meta-llama/Llama-3.1-8B-Instruct")
    
    # Load data
    classifier.load_sarcasm_datasets()
    
    # Perform classification
    results = classifier.classify_sarcasm(num_examples=30000)
    
    # Save results
    classifier.save_results()
    
    logger.info("Classification completed successfully!")
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION RESULTS SUMMARY")
        print("="*60)
        print(f"Accuracy:  {results['accuracy']:.3f}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall:    {results['recall']:.3f}")
        print(f"F1 Score:  {results['f1_score']:.3f}")
        print(f"Correct:   {results['correct_predictions']}/{results['total_examples']}")
        
        print("\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"           Predicted 0  Predicted 1")
        print(f"Actual 0     {cm[0,0]:^10}    {cm[0,1]:^10}")
        print(f"Actual 1     {cm[1,0]:^10}    {cm[1,1]:^10}")

if __name__ == "__main__":
    main()