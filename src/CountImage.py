import os

def check_data_structure(data_path):
    for subset in ["training_set", "test_set"]:
        subset_path = os.path.join(data_path, subset)
        print(f"\nChecking subset: {subset}")
        for category in os.listdir(subset_path):  # cats, dogs
            category_path = os.path.join(subset_path, category)
            if os.path.isdir(category_path):
                num_files = len(os.listdir(category_path))
                print(f" - Category: {category} -> Number of images: {num_files}")
            else:
                print(f" - {category} is not a directory!")

data_path = "data/raw/"
check_data_structure(data_path)
