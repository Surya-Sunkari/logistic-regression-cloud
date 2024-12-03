import random

# Function to split the data into training and test sets
def split_data(input_file, train_file, test_file, train_ratio=0.8, seed=42):
    # Read the input file
    with open(input_file, 'r') as file:
        data = file.readlines()
    
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Determine the split index
    split_index = int(len(data) * train_ratio)
    
    # Split the data
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Write the training data to the train file
    with open(train_file, 'w') as train_file:
        train_file.writelines(train_data)
    
    # Write the test data to the test file
    with open(test_file, 'w') as test_file:
        test_file.writelines(test_data)

# Example usage:
input_file_path = 'SmallTrainingData.txt'  # Path to the input data file
train_file_path = 'train_data.txt'         # Path to the output train file
test_file_path = 'test_data.txt'           # Path to the output test file

split_data(input_file_path, train_file_path, test_file_path)

# print(f"Data split complete. Training set saved to {train_file_path}, Test set saved to {test_file_path}.")
