import torch
import argparse
from models.bcm_transformer import BCMTransformerForSequenceClassification
from data.dataset import tokenize_and_prepare_imdb
from utils.train_utils import train_model, evaluate_model, analyze_compression

def main():
    parser = argparse.ArgumentParser(description='BCM Transformer for IMDB Sentiment Analysis')
    parser.add_argument('--block_size', type=int, default=32, help='Block size for BCM compression')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'analyze'],
                        help='Operation mode')
    args = parser.parse_args()
    
    # Load dataset
    print("Loading and preparing the IMDB dataset...")
    train_dataloader, test_dataloader, vocab = tokenize_and_prepare_imdb(args.batch_size)
    
    # Create model
    model = BCMTransformerForSequenceClassification(
        vocab_size=len(vocab),
        num_classes=2,
        block_size=args.block_size,
        embed_dim=args.embed_dim
    )
    
    # Count parameters
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Execute based on mode
    if args.mode == 'train':
        # Train the model
        print("Training the model...")
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )
    
    elif args.mode == 'evaluate':
        # Load best model and evaluate
        model.load_state_dict(torch.load('best_bcm_transformer.pt'))
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        accuracy = evaluate_model(model, test_dataloader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    elif args.mode == 'analyze':
        # Analyze compression
        compression_ratio = analyze_compression(model)
        print(f"Overall Compression Ratio: {compression_ratio:.2f}x")

if __name__ == "__main__":
    main()
