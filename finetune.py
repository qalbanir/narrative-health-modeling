import random
import numpy as np
import pandas as pd
import os
import joblib
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from simpletransformers.classification import ClassificationArgs
from simpletransformers.classification import ClassificationModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import logging
import contractions
import openpyxl
from tqdm import tqdm
from datetime import datetime
from IPython.display import display, Javascript

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Read in training data
train_data = pd.read_json("/path/to/project/Active/TD/SPAN/psm_combined.json", lines=True)

# Preprocess and standardize labels
train_data = train_data[train_data['PNEOA_scaled'] != '.']
scaler = StandardScaler()
train_data['PNEOA_scaled_new'] = scaler.fit_transform(train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
train_data = train_data[train_data['text'] != '']
train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))

train_data_df = train_data[['PARTID', 'text', 'PNEOA_scaled_new']].rename(columns={
    'PNEOA_scaled_new': 'labels'
})

# Extract text and labels for model input
train_texts = train_data_df['text'].astype(str).tolist()
train_labels = train_data_df['labels'].astype(float).tolist()

print(train_data_df.head())
print(len(train_texts), len(train_labels))

output_dir = "/path/to/project/Active/TD/SPAN/apsmc/Modelapsmc/"
ffn_models_dir = output_dir + "ffn_models/"
embeddings_dir = output_dir + "embeddings/"

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ffn_models_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# Save the scaler for later use
scaler_path = output_dir + 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Logging setup
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Custom function to calculate Pearson correlation
def pearson_correlation(y_true, y_pred):
    """Calculate Pearson correlation coefficient and p-value."""
    correlation, p_value = pearsonr(y_true, y_pred)
    return correlation, p_value

def flatten_outputs(model_outputs):
    """Flatten SimpleTransformers/transformers outputs into a 1D list of floats."""
    if isinstance(model_outputs, list):
        preds = []
        for output in model_outputs:
            if isinstance(output, (list, tuple, np.ndarray)):
                if hasattr(output, 'flatten'):
                    arr = output.flatten()
                    if len(arr) > 0:
                        preds.append(float(arr[0]))
                else:
                    # list/tuple
                    preds.append(float(output[0]))
            else:
                preds.append(float(output))
    else:
        arr = np.array(model_outputs).flatten()
        preds = [float(v) for v in arr]
    return preds


# Define function to extract embeddings from layer 23
def get_layer_embeddings(model, tokenizer, texts, layer_num=23):
    device = model.device  # Get the device model is currently on
    print("Device")
    print(device)

    cls_embeddings = []
    for text in texts:
        c_emb = []
        for window in tqdm(text, desc="Processing Windows"):
            # 1. Tokenize the window and assign it to the 'inputs' variable
            inputs = tokenizer(window, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # 2. Move the tokenized inputs to the correct device
            inputs = inputs.to(device)
            
            # 3. Get model outputs
            with torch.no_grad(): # Use no_grad for inference to save memory and compute
                outputs = model(**inputs, output_hidden_states=True)

            layer_embeddings = outputs.hidden_states[layer_num]
            last_embeddings = layer_embeddings[:, 0, :].detach().cpu().numpy()
            c_emb.append(last_embeddings.squeeze())
            
        cls_embeddings.append(c_emb)

    return cls_embeddings

# Sliding window size and overlap
window_size = 500
overlap = 100

# Function to generate sliding window samples
def create_sliding_windows(text, window_size, overlap):
    windows = []
    text=text.split()
    for i in range(0, len(text), overlap):
        window = text[i:i+window_size]
        window = " ".join(window)
        windows.append(window)
    return windows

# Define the Feedforward Neural Network with Dropout
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# Set hyperparameters 
hidden_size1 = 256  
hidden_size2 = 128 
learning_rate = 1e-3  
num_epochs = 50      
weight_decay = 1e-4 

# Cross-validation with best model tracking
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Results storage
finetuning_results = []
ffn_results = []
partitions = []

race_results_all = {"Transformer": [], "FFN": []}

def evaluate_by_race(df, preds, race_col="PRACE", label_col="labels", prefix="Transformer"):
    df = df.copy()
    df["preds"] = preds
    results = {}
    for race_value, race_name in [(1, "White"), (2, "Black")]:
        subset = df[df[race_col] == race_value]
        if len(subset) == 0:
            continue
        true_vals = subset[label_col].values
        pred_vals = subset["preds"].values
        mse = np.mean((true_vals - pred_vals) ** 2)
        mae = np.mean(np.abs(true_vals - pred_vals))
        r2 = r2_score(true_vals, pred_vals) if len(subset) > 1 else np.nan
        corr = np.corrcoef(true_vals, pred_vals)[0, 1] if len(subset) > 1 else np.nan
        results[race_name] = {
            "N": len(subset),
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4) if not np.isnan(r2) else None,
            "Corr": round(corr, 4) if not np.isnan(corr) else None
        }

    print(f"\n===== {prefix} Results by Race =====")
    for race, metrics in results.items():
        print(f"{race}: N={metrics['N']}, MSE={metrics['MSE']}, MAE={metrics['MAE']}, R²={metrics['R2']}, Corr={metrics['Corr']}")
    print("===================================\n")
    if prefix not in race_results_all:
        race_results_all[prefix] = []
    race_results_all[prefix].append(results)

def summarize_race_results():
    print("\n======= AVERAGE RESULTS BY RACE ACROSS ALL FOLDS =======")
    for model_type in ["Transformer", "FFN"]:
        print(f"\n--- {model_type} ---")
        for race in ["White", "Black"]:
            vals = [r.get(race, {}) for r in race_results_all[model_type] if r.get(race)]
            if not vals: continue
            avg_mse = np.mean([v["MSE"] for v in vals])
            avg_mae = np.mean([v["MAE"] for v in vals])
            avg_r2 = np.nanmean([v["R2"] for v in vals])
            avg_corr = np.nanmean([v["Corr"] for v in vals])
            print(f"{race}: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}, Corr={avg_corr:.4f}")

# Lists for tracking all fold results
all_ft_r2 = []
all_ft_mse = []
all_ft_mae = []
all_ft_pearson_r = []
all_ft_pearson_p = []

all_ffn_train_mae = []
all_ffn_train_mse = []
all_ffn_train_r2 = []
all_ffn_train_corr = []
all_ffn_test_mae = []
all_ffn_test_mse = []
all_ffn_test_r2 = []
all_ffn_test_corr = []

fold_num = 0
for train_index, test_index in kf.split(train_data_df):
    fold_start_time = datetime.now()
    fold_num += 1
    print(f"\n{'='*60}")
    print(f"STARTING FOLD {fold_num}/5")
    print(f"{'='*60}")
    
    training_data_df = train_data_df.iloc[train_index]
    test_data = train_data_df.iloc[test_index]

    # Initialize training and validation data
    ids = random.sample(list(training_data_df["PARTID"]), int(0.95 * (training_data_df.shape[0])))
    training_data = training_data_df[training_data_df["PARTID"].isin(ids)]
    validation_data = training_data_df[~training_data_df["PARTID"].isin(ids)]
    print("training_data: ", training_data.shape)
    print("validation_data: ", validation_data.shape)
    print("test_data: ", test_data.shape)

    # Save PARTIDs for this fold in a flattened format
    for pid in training_data["PARTID"].tolist():
        partitions.append({"fold": fold_num, "PARTID": pid, "set": "train"})
    for pid in validation_data["PARTID"].tolist():
        partitions.append({"fold": fold_num, "PARTID": pid, "set": "validation"})
    for pid in test_data["PARTID"].tolist():
        partitions.append({"fold": fold_num, "PARTID": pid, "set": "test"})
    
    # =================================================================
    # FINE-TUNING PHASE
    # =================================================================
    print(f"\n{'='*40}")
    print(f"FINE-TUNING FOLD {fold_num}")
    print(f"{'='*40}")
    
    # Define model arguments - each fold gets its own directory
    fold_output_dir = output_dir + f"fold_{fold_num}/"
    model_args = ClassificationArgs(
        output_dir=fold_output_dir,
        sliding_window=True,
        use_early_stopping=True,
        early_stopping_metric="r2",
        early_stopping_metric_minimize=False,
        early_stopping_patience=5,
        num_train_epochs=20,
        learning_rate=2e-5,
        evaluate_during_training=True,
        regression=True,
        train_batch_size=16,
        eval_batch_size=8,
        evaluate_during_training_steps=1000,
        max_seq_length=512,
        no_cache=True,
        overwrite_output_dir=True,
        reprocess_input_data=True,
        gradient_accumulation_steps=2,
        save_best_model=True,
        save_model_every_epoch=False,    
        save_eval_checkpoints=False,     
        save_optimizer_and_scheduler=False,   
        best_model_dir=fold_output_dir + "best_model/", 
        save_steps=-1,                      # No intermediate saves
        no_save=False,                      # Still save the final best model
        silent=True,
        dataloader_num_workers=0, 
        use_multiprocessing=False,               # Enable multiprocessing for training
        use_multiprocessing_for_evaluation=False,# Enable multiprocessing for evaluation
    )

    # Initialize and train the model
    print(f"\nInitializing model for fold {fold_num}...")
    model = ClassificationModel(
        "roberta",
        "roberta-large",
        num_labels=1,
        args=model_args,
        use_cuda=True
    )

    print(f"Training model for fold {fold_num}...")
    model.train_model(training_data, eval_df=validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    
    # Evaluate the fine-tuned model
    print(f"Evaluating fine-tuned model for fold {fold_num}...")
    result, model_outputs, wrong_predictions = model.eval_model(test_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    
    # Calculate Pearson correlation manually
    y_true = test_data['labels'].tolist()
    # Handle model_outputs and ensure it's a flat list of numbers
#    if isinstance(model_outputs, list):
#        # If it's a list, flatten any nested structures
#        y_pred = []
#        for output in model_outputs:
#            if isinstance(output, (list, tuple, np.ndarray)):
#                # If output is itself a sequence, take the first element or flatten
#                if hasattr(output, '__len__') and len(output) > 0:
#                    y_pred.append(float(output.flatten()[0]) if hasattr(output, 'flatten') else float(output[0]))
#                else:
#                    y_pred.append(float(output))
#            else:
#                y_pred.append(float(output))
#    else:
#        y_pred = model_outputs.flatten().tolist()
    y_pred = flatten_outputs(model_outputs)

    pearson_r, pearson_p = pearson_correlation(y_true, y_pred)
    
    # Store fine-tuning results
    ft_fold_result = {
        'fold': fold_num,
        'r2': result['r2'],
        'mse': result['mse'],
        'mae': result['mae'],
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }
    finetuning_results.append(ft_fold_result)
    
    # Add to lists for averaging
    all_ft_r2.append(result['r2'])
    all_ft_mse.append(result['mse'])
    all_ft_mae.append(result['mae'])
    all_ft_pearson_r.append(pearson_r)
    all_ft_pearson_p.append(pearson_p)
    
    # Print fine-tuning fold summary
    print(f"\n{'='*30}")
    print(f"FINE-TUNING FOLD {fold_num} RESULTS")
    print(f"{'='*30}")
    print(f"R²:              {result['r2']:.4f}")
    print(f"MSE:             {result['mse']:.4f}")
    print(f"MAE:             {result['mae']:.4f}")
    print(f"Pearson r:       {pearson_r:.4f}")
    print(f"Pearson p-value: {pearson_p:.4f}")
    print(f"{'='*30}")
    
    # =========================================================
    # SAVE TRAIN & VALIDATION PREDICTIONS FOR FINE-TUNED MODEL
    # =========================================================
    # Train predictions
    train_outputs, _ = model.predict(training_data['text'].tolist())
    y_train_true = training_data['labels'].tolist()
    y_train_pred = flatten_outputs(train_outputs)

    ft_train_pred_df = pd.DataFrame({
        'PARTID': training_data['PARTID'].values,
        'true_label': y_train_true,
        'predicted_label': y_train_pred
    })
    ft_train_pred_path = os.path.join(output_dir, f'finetune_train_predictions_fold{fold_num}.csv')
    ft_train_pred_df.to_csv(ft_train_pred_path, index=False)
    print(f"Fine-tuned TRAIN predictions saved to: {ft_train_pred_path}")

    # Validation predictions
    val_outputs, _ = model.predict(validation_data['text'].tolist())
    y_val_true = validation_data['labels'].tolist()
    y_val_pred = flatten_outputs(val_outputs)

    ft_val_pred_df = pd.DataFrame({
        'PARTID': validation_data['PARTID'].values,
        'true_label': y_val_true,
        'predicted_label': y_val_pred
    })
    ft_val_pred_path = os.path.join(output_dir, f'finetune_val_predictions_fold{fold_num}.csv')
    ft_val_pred_df.to_csv(ft_val_pred_path, index=False)
    print(f"Fine-tuned VAL predictions saved to: {ft_val_pred_path}")


    # >>> NEW: save fine-tuned CV test predictions for this fold
    cv_pred_df = pd.DataFrame({
        'PARTID': test_data['PARTID'].values,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    cv_pred_path = os.path.join(output_dir, f'finetune_cv_predictions_fold{fold_num}.csv')
    cv_pred_df.to_csv(cv_pred_path, index=False)
    print(f"Fine-tune CV test predictions saved to: {cv_pred_path}")
    # <<< END NEW
    
    # Evaluate fine-tuned model by race
    # Make sure PRACE column exists for evaluation
    if "PRACE" not in test_data.columns and "PRACE" in train_data.columns:
        test_data["PRACE"] = train_data.loc[test_data.index, "PRACE"].values

    evaluate_by_race(test_data, y_pred, race_col="PRACE", label_col="labels", prefix="Transformer (Fine-tuned)")

    # =================================================================
    # SAVE AND RELOAD MODEL 
    # =================================================================
    print(f"\nSaving and reloading model for fold {fold_num}...")
    model_save_path = fold_output_dir + "CV_A/"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save using SimpleTransformers model's built-in methods
    model.model.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)
    model.config.save_pretrained(model_save_path)
    
    # Reload the model components
    model.model = model.model.from_pretrained(model_save_path)
    model.tokenizer = model.tokenizer.from_pretrained(model_save_path)
    model.config = model.config.from_pretrained(model_save_path)
    
    print(f"Model saved and reloaded from: {model_save_path}")
    
    # =================================================================
    # FFN TRAINING PHASE (using the reloaded model)
    # =================================================================
    print(f"\n{'='*40}")
    print(f"FFN TRAINING FOLD {fold_num}")
    print(f"{'='*40}")
    
    ffn_start_time = datetime.now()
    print(f"FFN training started at: {ffn_start_time}")
    
    # Use the model components from the SimpleTransformers model
    ft_model = model.model
    ft_tokenizer = model.tokenizer
    ft_config = model.config
    
    # Move model to GPU if not already
    ft_model = ft_model.to(device)
    print(f"Model moved to device: {ft_model.device}")
    
    # Use the same train/test splits as fine-tuning
    training_data_ffn = training_data.copy()
    test_data_ffn = test_data.copy()
    val_data_ffn = validation_data.copy()
    
    # Create sliding windows
    training_data_ffn['sliding_windows'] = training_data_ffn['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))
    test_data_ffn['sliding_windows'] = test_data_ffn['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))
    val_data_ffn['sliding_windows'] = val_data_ffn['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))

    all_texts_test = test_data_ffn['sliding_windows'].tolist()
    all_texts_val = val_data_ffn['sliding_windows'].tolist()
    all_texts_train = training_data_ffn['sliding_windows'].tolist()

    # Extract embeddings
    embed_start_time = datetime.now()
    print(f"Extracting test embeddings...")
    c_embeddings_test = get_layer_embeddings(ft_model, ft_tokenizer, all_texts_test)
    c_embeddings_val = get_layer_embeddings(ft_model, ft_tokenizer, all_texts_val)
    print("c_embeddings_test", len(c_embeddings_test))
    
    # Save test embeddings
    embeddings_save_path = os.path.join(embeddings_dir, f'test_embeddings_fold{fold_num}.pkl')
    with open(embeddings_save_path, "wb") as f:
        pickle.dump(c_embeddings_test, f)

    embeddings_save_path = os.path.join(embeddings_dir, f'val_embeddings_fold{fold_num}.pkl')
    with open(embeddings_save_path, "wb") as f:
        pickle.dump(c_embeddings_val, f)
    
    embed_mid_time = datetime.now()
    print(f"Test embedding extraction time: {embed_mid_time - embed_start_time}")
    print(f"Extracting train embeddings...")
    
    c_embeddings_train = get_layer_embeddings(ft_model, ft_tokenizer, all_texts_train)
    print("c_embeddings_train", len(c_embeddings_train))
    
    # Save train embeddings
    embeddings_save_path = os.path.join(embeddings_dir, f'train_embeddings_fold{fold_num}.pkl')
    with open(embeddings_save_path, "wb") as f:
        pickle.dump(c_embeddings_train, f)
    
    embed_end_time = datetime.now()
    print(f"Train embedding extraction time: {embed_end_time - embed_mid_time}")

    # Process embeddings (mean pooling)
    test_data_emb = np.empty((0, 1024))
    for i in range(len(c_embeddings_test)):
        x = np.asarray(c_embeddings_test[i])
        d = np.mean(x, axis=0)
        d = np.expand_dims(d, axis=0)
        test_data_emb = np.append(test_data_emb, d, axis=0)
        
    print("test_data_emb shape: ", test_data_emb.shape) 

    val_data_emb = np.empty((0, 1024))
    for i in range(len(c_embeddings_val)):
        x = np.asarray(c_embeddings_val[i])
        d = np.mean(x, axis=0)
        d = np.expand_dims(d, axis=0)
        val_data_emb = np.append(val_data_emb, d, axis=0)
        
    print("val_data_emb shape: ", val_data_emb.shape) 

    train_data_emb = np.empty((0, 1024))
    for i in range(len(c_embeddings_train)):
        x = np.asarray(c_embeddings_train[i])
        d = np.mean(x, axis=0)
        d = np.expand_dims(d, axis=0)
        train_data_emb = np.append(train_data_emb, d, axis=0)
        
    print("train_data_emb shape: ", train_data_emb.shape)  
    
    # Get labels for current fold
    train_labels_fold = training_data_ffn['labels'].astype(float).tolist()
    test_labels_fold = test_data_ffn['labels'].astype(float).tolist()
    val_labels_fold = val_data_ffn['labels'].astype(float).tolist()
     
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.tensor(train_data_emb, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(test_data_emb, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(val_data_emb, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(train_labels_fold, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(test_labels_fold, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(val_labels_fold, dtype=torch.float32).to(device)

    # Set input size based on training set
    input_size = X_train_tensor.shape[1]

    # Initialize the FFN model, loss function, optimizer, and scheduler for this fold
    ffn_model = FeedforwardNN(input_size, hidden_size1, hidden_size2, 1).to(device)
    print(f"FFN model moved to device: {ffn_model.device if hasattr(ffn_model, 'device') else 'N/A'}")
    criterion = nn.L1Loss()  # L1Loss for MAE
    optimizer = optim.Adam(ffn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Lists for tracking losses
    train_losses = []
    test_losses = []
    val_losses = []
    min_val=5000
    # Training loop
    for epoch in range(num_epochs):
        ffn_model.train()
        optimizer.zero_grad()
        outputs = ffn_model(X_train_tensor).squeeze()
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)
        train_losses.append(train_loss.item())

        ffn_model.eval()
        with torch.no_grad():
            val_outputs = ffn_model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
    
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {val_loss.item():.4f}")
            if (val_loss.item()>=min_val):
                print("\n STOP TRAINING")
                break
            else:
                min_val=val_loss.item()
            
    
    # Final evaluation on the training and test sets
    ffn_model.eval()
    with torch.no_grad():
        train_pred = ffn_model(X_train_tensor).squeeze().cpu().numpy()
        test_pred = ffn_model(X_test_tensor).squeeze().cpu().numpy()
        # >>> NEW: also get validation predictions
        val_pred = ffn_model(X_val_tensor).squeeze().cpu().numpy()
        # <<< END NEW
    
    y_train_np = y_train_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    y_val_np = y_val_tensor.cpu().numpy()
    
    # Calculate metrics for training set
    mae_train = mean_absolute_error(y_train_np, train_pred)
    mse_train = mean_squared_error(y_train_np, train_pred)
    r2_train = r2_score(y_train_np, train_pred)
    corr_train = np.corrcoef(y_train_np, train_pred)[0, 1]
    
    # Calculate metrics for test set
    mae_test = mean_absolute_error(y_test_np, test_pred)
    mse_test = mean_squared_error(y_test_np, test_pred)
    r2_test = r2_score(y_test_np, test_pred)
    corr_test = np.corrcoef(y_test_np, test_pred)[0, 1]
    
    # >>> NEW: save FFN predictions (train/val/test) for this fold
    ffn_train_pred_df = pd.DataFrame({
        'PARTID': training_data_ffn['PARTID'].values,
        'true_label': y_train_np,
        'predicted_label': train_pred
    })
    ffn_val_pred_df = pd.DataFrame({
        'PARTID': val_data_ffn['PARTID'].values,
        'true_label': y_val_np,
        'predicted_label': val_pred
    })
    ffn_test_pred_df = pd.DataFrame({
        'PARTID': test_data_ffn['PARTID'].values,
        'true_label': y_test_np,
        'predicted_label': test_pred
    })
    
    ffn_train_pred_path = os.path.join(output_dir, f'ffn_train_predictions_fold{fold_num}.csv')
    ffn_val_pred_path = os.path.join(output_dir, f'ffn_val_predictions_fold{fold_num}.csv')
    ffn_test_pred_path = os.path.join(output_dir, f'ffn_test_predictions_fold{fold_num}.csv')
    
    ffn_train_pred_df.to_csv(ffn_train_pred_path, index=False)
    ffn_val_pred_df.to_csv(ffn_val_pred_path, index=False)
    ffn_test_pred_df.to_csv(ffn_test_pred_path, index=False)
    
    print(f"FFN train predictions saved to: {ffn_train_pred_path}")
    print(f"FFN val predictions saved to:   {ffn_val_pred_path}")
    print(f"FFN test predictions saved to:  {ffn_test_pred_path}")
    # <<< END NEW
    
    # Store FFN results for this fold
    ffn_fold_results = {
        'fold': fold_num,
        'train_mae': mae_train,
        'train_mse': mse_train,
        'train_r2': r2_train,
        'train_corr': corr_train,
        'test_mae': mae_test,
        'test_mse': mse_test,
        'test_r2': r2_test,
        'test_corr': corr_test,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1]
    }
    ffn_results.append(ffn_fold_results)
    
    # Add to lists for averaging
    all_ffn_train_mae.append(mae_train)
    all_ffn_train_mse.append(mse_train)
    all_ffn_train_r2.append(r2_train)
    all_ffn_train_corr.append(corr_train)
    all_ffn_test_mae.append(mae_test)
    all_ffn_test_mse.append(mse_test)
    all_ffn_test_r2.append(r2_test)
    all_ffn_test_corr.append(corr_test)
    
    # Save the trained FFN model for this fold
    ffn_model_path = os.path.join(ffn_models_dir, f'feedforward_nn_model_fold{fold_num}.pth')
    torch.save(ffn_model.state_dict(), ffn_model_path)
    
    print(f"\n{'='*30}")
    print(f"FFN FOLD {fold_num} RESULTS")
    print(f"{'='*30}")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Val Loss: {val_losses[-1]:.4f}")
    print(f"Train MAE: {mae_train:.4f}, MSE: {mse_train:.4f}, R²: {r2_train:.4f}, Correlation: {corr_train:.4f}")
    print(f"Test MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, R²: {r2_test:.4f}, Correlation: {corr_test:.4f}")
    print(f"FFN model saved to: {ffn_model_path}")
    
    # Evaluate FFN predictions by race
    if "PRACE" not in test_data_ffn.columns and "PRACE" in train_data.columns:
        test_data_ffn["PRACE"] = train_data.loc[test_data_ffn.index, "PRACE"].values

    evaluate_by_race(test_data_ffn, test_pred, race_col="PRACE", label_col="labels", prefix="FFN (ftffnammc)")

    fold_end_time = datetime.now()
    print(f"Total fold time: {fold_end_time - fold_start_time}")
    print(f"{'='*30}")

# =================================================================
# FINAL RESULTS SUMMARY
# =================================================================
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")

# Fine-tuning results summary
print(f"\n{'='*50}")
print("FINE-TUNING RESULTS SUMMARY")
print(f"{'='*50}")

ft_results_df = pd.DataFrame(finetuning_results)
ft_avg_results = ft_results_df.mean()
ft_std_results = ft_results_df.std()

print(f"Average R²:         {ft_avg_results['r2']:.4f} ± {ft_std_results['r2']:.4f}")
print(f"Average MSE:        {ft_avg_results['mse']:.4f} ± {ft_std_results['mse']:.4f}")
print(f"Average MAE:        {ft_avg_results['mae']:.4f} ± {ft_std_results['mae']:.4f}")
print(f"Average Pearson r:  {ft_avg_results['pearson_r']:.4f} ± {ft_std_results['pearson_r']:.4f}")
print(f"Average Pearson p:  {ft_avg_results['pearson_p']:.4f} ± {ft_std_results['pearson_p']:.4f}")

# FFN results summary
print(f"\n{'='*50}")
print("FFN RESULTS SUMMARY")
print(f"{'='*50}")

# Calculate average metrics across all folds
avg_train_mae = np.mean(all_ffn_train_mae)
avg_train_mse = np.mean(all_ffn_train_mse)
avg_train_r2 = np.mean(all_ffn_train_r2)
avg_train_corr = np.mean(all_ffn_train_corr)
avg_test_mae = np.mean(all_ffn_test_mae)
avg_test_mse = np.mean(all_ffn_test_mse)
avg_test_r2 = np.mean(all_ffn_test_r2)
avg_test_corr = np.mean(all_ffn_test_corr)

# Calculate standard deviations
std_train_mae = np.std(all_ffn_train_mae)
std_train_mse = np.std(all_ffn_train_mse)
std_train_r2 = np.std(all_ffn_train_r2)
std_train_corr = np.std(all_ffn_train_corr)
std_test_mae = np.std(all_ffn_test_mae)
std_test_mse = np.std(all_ffn_test_mse)
std_test_r2 = np.std(all_ffn_test_r2)
std_test_corr = np.std(all_ffn_test_corr)

print(f"Average Training Metrics (Mean ± Std):")
print(f"MAE: {avg_train_mae:.4f} ± {std_train_mae:.4f}")
print(f"MSE: {avg_train_mse:.4f} ± {std_train_mse:.4f}")
print(f"R²:  {avg_train_r2:.4f} ± {std_train_r2:.4f}")
print(f"Correlation: {avg_train_corr:.4f} ± {std_train_corr:.4f}")

print(f"\nAverage Test Metrics (Mean ± Std):")
print(f"MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
print(f"MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
print(f"R²:  {avg_test_r2:.4f} ± {std_test_r2:.4f}")
print(f"Correlation: {avg_test_corr:.4f} ± {std_test_corr:.4f}")

# Save all results
print(f"\n{'='*50}")
print("SAVING RESULTS")
print(f"{'='*50}")

# Save PARTID splits per fold
partitions_df = pd.DataFrame(partitions)
partitions_df.to_json(output_dir + "fold_partitions.json", orient="records", indent=2)
partitions_df.to_csv(output_dir + "fold_partitions.csv", index=False)

# Save fine-tuning results
ft_results_df.to_excel(output_dir + 'finetuning_results.xlsx', index=False)
ft_results_df.to_csv(output_dir + 'finetuning_results.csv', index=False)

# Save FFN results
ffn_results_df = pd.DataFrame(ffn_results)
ffn_results_df.to_excel(output_dir + 'ffn_results.xlsx', index=False)
ffn_results_df.to_csv(output_dir + 'ffn_results.csv', index=False)

# Save all results to pickle files
finetuning_results_path = os.path.join(output_dir, 'finetuning_results.pkl')
with open(finetuning_results_path, 'wb') as f:
    pickle.dump(finetuning_results, f)

ffn_results_path = os.path.join(output_dir, 'ffn_results.pkl')
with open(ffn_results_path, 'wb') as f:
    pickle.dump(ffn_results, f)

# Save summary statistics
summary_stats = {
    'finetuning': {
        'avg_r2': ft_avg_results['r2'], 'std_r2': ft_std_results['r2'],
        'avg_mse': ft_avg_results['mse'], 'std_mse': ft_std_results['mse'],
        'avg_mae': ft_avg_results['mae'], 'std_mae': ft_std_results['mae'],
        'avg_pearson_r': ft_avg_results['pearson_r'], 'std_pearson_r': ft_std_results['pearson_r'],
        'avg_pearson_p': ft_avg_results['pearson_p'], 'std_pearson_p': ft_std_results['pearson_p']
    },
    'ffn': {
        'avg_train_mae': avg_train_mae, 'std_train_mae': std_train_mae,
        'avg_train_mse': avg_train_mse, 'std_train_mse': std_train_mse,
        'avg_train_r2': avg_train_r2, 'std_train_r2': std_train_r2,
        'avg_train_corr': avg_train_corr, 'std_train_corr': std_train_corr,
        'avg_test_mae': avg_test_mae, 'std_test_mae': std_test_mae,
        'avg_test_mse': avg_test_mse, 'std_test_mse': std_test_mse,
        'avg_test_r2': avg_test_r2, 'std_test_r2': std_test_r2,
        'avg_test_corr': avg_test_corr, 'std_test_corr': std_test_corr
    }
}

summary_path = os.path.join(output_dir, 'summary_statistics.pkl')
with open(summary_path, 'wb') as f:
    pickle.dump(summary_stats, f)

print(f"Fine-tuning results saved to: {finetuning_results_path}")
print(f"FFN results saved to: {ffn_results_path}")
print(f"Summary statistics saved to: {summary_path}")
print(f"All FFN models saved in: {ffn_models_dir}")
print(f"All embeddings saved in: {embeddings_dir}")
print(f"Fold partitions saved to: {output_dir}fold_partitions.json and .csv")

summarize_race_results()

print(f"\n{'='*70}")
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")

# Save the race-wise average results across folds
race_summary_path = os.path.join(output_dir, "race_results_summary.pkl")
with open(race_summary_path, "wb") as f:
    pickle.dump(race_results_all, f)

# Also export as CSV for convenience
for model_type in race_results_all:
    records = []
    for fold_idx, fold_results in enumerate(race_results_all[model_type], 1):
        for race, metrics in fold_results.items():
            records.append({
                "Model": model_type,
                "Fold": fold_idx,
                "Race": race,
                **metrics
            })
    race_df = pd.DataFrame(records)
    race_df.to_csv(os.path.join(output_dir, f"{model_type.replace(' ', '_').lower()}_race_results.csv"), index=False)
    print(f"Race results for {model_type} saved to {output_dir}{model_type.replace(' ', '_').lower()}_race_results.csv")


# ===========================
# Average Results by Race
# ===========================
print("\n======= AVERAGE RESULTS BY RACE ACROSS ALL FOLDS =======")
average_race_results = {}  # store for saving

for model_type in ["Transformer (Fine-tuned)", "FFN (ftffnammc)"]:
    print(f"\n--- {model_type} ---")
    average_race_results[model_type] = {}
    for race in ["White", "Black"]:
        vals = [r.get(race, {}) for r in race_results_all.get(model_type, []) if r.get(race)]
        if not vals:
            continue
        avg_mse = np.mean([v["MSE"] for v in vals])
        avg_mae = np.mean([v["MAE"] for v in vals])
        avg_r2 = np.nanmean([v["R2"] for v in vals])
        avg_corr = np.nanmean([v["Corr"] for v in vals])
        print(f"{race}: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}, Corr={avg_corr:.4f}")

        # Save in dict for pickle/CSV
        average_race_results[model_type][race] = {
            "MSE": avg_mse,
            "MAE": avg_mae,
            "R2": avg_r2,
            "Corr": avg_corr
        }

# Save average metrics by race
average_race_path_pkl = os.path.join(output_dir, 'average_race_results.pkl')
with open(average_race_path_pkl, 'wb') as f:
    pickle.dump(average_race_results, f)

average_race_path_csv = os.path.join(output_dir, 'average_race_results.csv')
df_avg_race = pd.DataFrame([
    {"Model": model, "Race": race, **metrics} 
    for model, races in average_race_results.items()
    for race, metrics in races.items()
])
df_avg_race.to_csv(average_race_path_csv, index=False)

print(f"\nAverage results by race saved to:\n- {average_race_path_pkl}\n- {average_race_path_csv}")
