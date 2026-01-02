"""
TWO-STAGE IDENTIFICATION SYSTEM
================================
Stage 1: Get Top-K candidates using classifier
Stage 2: Verify each candidate using embedding similarity

This approach maximizes accuracy with limited data.

Expected Results:
- Top-1 Classification: 55-65%
- Top-5 Classification: 85-90%
- Top-10 Classification: 92-95%
- Two-Stage Final: 85-90%
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)


class TwoStageIdentifier:
    """
    Two-stage identification system.
    
    Stage 1: Classification to get top-K candidates
    Stage 2: Embedding verification to pick best match
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = "cpu",
        top_k: int = 10
    ):
        self.device = device
        self.top_k = top_k
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_to_name = {int(k): v for k, v in self.metadata['label_to_name'].items()}
        self.name_to_label = self.metadata['name_to_label']
        self.num_classes = self.metadata['num_classes']
        
        # Enrollment database (embeddings)
        self.enrolled_embeddings = {}
        self.enrolled_names = []
        
        logger.info(f"Loaded model with {self.num_classes} classes")
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        from models.networks import PalmVeinClassifier
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        num_classes = checkpoint['num_classes']
        
        model = PalmVeinClassifier(
            num_classes=num_classes,
            backbone=config.get('backbone', 'lightweight'),
            pretrained=False,
            embedding_dim=config.get('embedding_dim', 256),
            dropout=0.0  # No dropout during inference
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path: str, is_ir: bool = False) -> torch.Tensor:
        """Load and preprocess image."""
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        # Convert IR to grayscale
        if is_ir and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # To tensor
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img).unsqueeze(0)
    
    def get_embedding(self, rgb_path: str, ir_path: str) -> np.ndarray:
        """Get embedding for an image pair."""
        rgb = self.preprocess_image(rgb_path, is_ir=False).to(self.device)
        ir = self.preprocess_image(ir_path, is_ir=True).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_embedding(rgb, ir)
        
        return embedding.cpu().numpy().flatten()
    
    def get_predictions(self, rgb_path: str, ir_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get classification predictions and embedding."""
        rgb = self.preprocess_image(rgb_path, is_ir=False).to(self.device)
        ir = self.preprocess_image(ir_path, is_ir=True).to(self.device)
        
        with torch.no_grad():
            output = self.model(rgb, ir)
            logits = output['logits']
            embedding = output['embeddings']
            
            probs = F.softmax(logits, dim=1)
        
        return probs.cpu().numpy().flatten(), embedding.cpu().numpy().flatten()
    
    def enroll(self, name: str, rgb_path: str, ir_path: str):
        """Enroll a person into the database."""
        embedding = self.get_embedding(rgb_path, ir_path)
        
        if name in self.enrolled_embeddings:
            # Average with existing embedding
            old_emb = self.enrolled_embeddings[name]
            self.enrolled_embeddings[name] = (old_emb + embedding) / 2
            self.enrolled_embeddings[name] /= np.linalg.norm(self.enrolled_embeddings[name])
        else:
            self.enrolled_embeddings[name] = embedding
            self.enrolled_names.append(name)
        
        logger.info(f"Enrolled: {name}")
    
    def enroll_from_dataset(self, data_dir: str, use_val: bool = True):
        """Enroll all persons from dataset."""
        from data.dataset import PalmVeinDataset
        
        dataset = PalmVeinDataset(data_dir, mode='val')
        
        for sample in dataset.samples:
            name = sample['name']
            rgb_path = str(sample['rgb_path'])
            ir_path = str(sample['ir_path'])
            
            self.enroll(name, rgb_path, ir_path)
        
        logger.info(f"Enrolled {len(self.enrolled_names)} persons from dataset")
    
    def identify_stage1(self, rgb_path: str, ir_path: str) -> List[Tuple[str, float]]:
        """
        Stage 1: Get top-K candidates from classifier.
        
        Returns:
            List of (name, probability) tuples
        """
        probs, _ = self.get_predictions(rgb_path, ir_path)
        
        # Get top-K indices
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]
        
        candidates = []
        for idx in top_k_indices:
            name = self.label_to_name.get(idx, f"unknown_{idx}")
            prob = probs[idx]
            candidates.append((name, float(prob)))
        
        return candidates
    
    def identify_stage2(
        self,
        rgb_path: str,
        ir_path: str,
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, float]]:
        """
        Stage 2: Verify candidates using embedding similarity.
        
        Returns:
            List of (name, classification_prob, similarity) tuples
        """
        query_embedding = self.get_embedding(rgb_path, ir_path)
        
        results = []
        for name, prob in candidates:
            if name in self.enrolled_embeddings:
                enrolled_emb = self.enrolled_embeddings[name]
                similarity = float(np.dot(query_embedding, enrolled_emb))
            else:
                similarity = 0.0
            
            results.append((name, prob, similarity))
        
        # Sort by combined score (weighted average)
        results.sort(key=lambda x: 0.3 * x[1] + 0.7 * x[2], reverse=True)
        
        return results
    
    def identify(
        self,
        rgb_path: str,
        ir_path: str,
        threshold: float = 0.5
    ) -> Tuple[str, float, Dict]:
        """
        Complete two-stage identification.
        
        Returns:
            best_match: Name of identified person (or "unknown")
            confidence: Confidence score
            details: Dictionary with stage details
        """
        # Stage 1: Get candidates
        candidates = self.identify_stage1(rgb_path, ir_path)
        
        # Stage 2: Verify candidates
        if self.enrolled_embeddings:
            results = self.identify_stage2(rgb_path, ir_path, candidates)
        else:
            results = [(name, prob, 0.0) for name, prob in candidates]
        
        # Best match
        if results:
            best_name, best_prob, best_sim = results[0]
            confidence = 0.3 * best_prob + 0.7 * best_sim if best_sim > 0 else best_prob
            
            if confidence < threshold:
                best_name = "unknown"
        else:
            best_name = "unknown"
            confidence = 0.0
        
        details = {
            'stage1_candidates': candidates,
            'stage2_results': results,
            'top1_classification': candidates[0] if candidates else None,
            'threshold': threshold
        }
        
        return best_name, confidence, details
    
    def verify(
        self,
        rgb_path: str,
        ir_path: str,
        claimed_name: str,
        threshold: float = 0.6
    ) -> Tuple[bool, float, Dict]:
        """
        Verify if person matches claimed identity.
        
        Returns:
            is_match: True if verified
            similarity: Similarity score
            details: Dictionary with details
        """
        if claimed_name not in self.enrolled_embeddings:
            return False, 0.0, {'error': f'{claimed_name} not enrolled'}
        
        query_embedding = self.get_embedding(rgb_path, ir_path)
        enrolled_embedding = self.enrolled_embeddings[claimed_name]
        
        similarity = float(np.dot(query_embedding, enrolled_embedding))
        is_match = similarity >= threshold
        
        details = {
            'similarity': similarity,
            'threshold': threshold,
            'claimed_name': claimed_name
        }
        
        return is_match, similarity, details
    
    def evaluate(self, data_dir: str, val_split: float = 0.2) -> Dict:
        """
        Evaluate the two-stage system.
        
        Returns:
            Dictionary with evaluation metrics
        """
        from data.dataset import PalmVeinDataset
        import random
        
        dataset = PalmVeinDataset(data_dir, mode='val')
        
        # Split into enrollment and test
        indices = list(range(len(dataset.samples)))
        random.shuffle(indices)
        split = int(len(indices) * (1 - val_split))
        
        enroll_indices = indices[:split]
        test_indices = indices[split:]
        
        # Enroll
        self.enrolled_embeddings = {}
        self.enrolled_names = []
        
        for idx in enroll_indices:
            sample = dataset.samples[idx]
            self.enroll(sample['name'], str(sample['rgb_path']), str(sample['ir_path']))
        
        # Test
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        twostage_correct = 0
        verify_correct = 0
        total = 0
        
        for idx in test_indices:
            sample = dataset.samples[idx]
            true_name = sample['name']
            rgb_path = str(sample['rgb_path'])
            ir_path = str(sample['ir_path'])
            
            # Classification accuracy
            candidates = self.identify_stage1(rgb_path, ir_path)
            candidate_names = [c[0] for c in candidates]
            
            if candidate_names and candidate_names[0] == true_name:
                top1_correct += 1
            if true_name in candidate_names[:5]:
                top5_correct += 1
            if true_name in candidate_names[:10]:
                top10_correct += 1
            
            # Two-stage accuracy
            best_name, _, _ = self.identify(rgb_path, ir_path, threshold=0.3)
            if best_name == true_name:
                twostage_correct += 1
            
            # Verification accuracy
            if true_name in self.enrolled_embeddings:
                is_match, _, _ = self.verify(rgb_path, ir_path, true_name, threshold=0.5)
                if is_match:
                    verify_correct += 1
            
            total += 1
        
        results = {
            'total_samples': total,
            'enrolled': len(self.enrolled_names),
            'top1_accuracy': 100 * top1_correct / total if total > 0 else 0,
            'top5_accuracy': 100 * top5_correct / total if total > 0 else 0,
            'top10_accuracy': 100 * top10_correct / total if total > 0 else 0,
            'twostage_accuracy': 100 * twostage_correct / total if total > 0 else 0,
            'verification_accuracy': 100 * verify_correct / total if total > 0 else 0
        }
        
        return results
    
    def save_database(self, path: str):
        """Save enrollment database."""
        data = {
            'enrolled_names': self.enrolled_names,
            'embeddings': {name: emb.tolist() for name, emb in self.enrolled_embeddings.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Database saved to {path}")
    
    def load_database(self, path: str):
        """Load enrollment database."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.enrolled_names = data['enrolled_names']
        self.enrolled_embeddings = {name: np.array(emb) for name, emb in data['embeddings'].items()}
        logger.info(f"Database loaded: {len(self.enrolled_names)} persons")


def main():
    """Test the two-stage system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Stage Identification System")
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata')
    parser.add_argument('--data_dir', type=str, default='final_folder', help='Dataset directory')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
    args = parser.parse_args()
    
    # Initialize
    identifier = TwoStageIdentifier(
        model_path=args.model,
        metadata_path=args.metadata,
        device='cpu',
        top_k=10
    )
    
    if args.evaluate:
        print("\nEvaluating Two-Stage System...")
        results = identifier.evaluate(args.data_dir)
        
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total test samples: {results['total_samples']}")
        print(f"Enrolled persons: {results['enrolled']}")
        print("-" * 50)
        print(f"Top-1 Accuracy:  {results['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy:  {results['top5_accuracy']:.2f}%")
        print(f"Top-10 Accuracy: {results['top10_accuracy']:.2f}%")
        print(f"Two-Stage Accuracy: {results['twostage_accuracy']:.2f}%")
        print(f"Verification Accuracy: {results['verification_accuracy']:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
