import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(csv_path, names=columns)

    data['center'] = data['center'].astype(str).str.strip()
    data['left'] = data['left'].astype(str).str.strip()
    data['right'] = data['right'].astype(str).str.strip()

    return data


def fix_center_paths(data, img_folder):
    fixed_paths = []

    for path in data['center']:
        file_name = os.path.basename(path)
        full_path = os.path.join(img_folder, file_name)
        fixed_paths.append(full_path)

    data['center'] = fixed_paths
    return data


def balance_data(data):
    
    straight_data = data[abs(data['steering']) < 0.05]
    turning_data = data[abs(data['steering']) >= 0.05]

    straight_data = straight_data.sample(frac=0.10, random_state=42)

    balanced_data = pd.concat([straight_data, turning_data])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data


def show_basic_info(title, data):
    print(f"\n{title}")
    print("Total rows:", len(data))
    print(data[['center', 'steering']].head())


def plot_histograms(original_data, balanced_data):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(original_data['steering'], bins=31)
    plt.title("Original Steering Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(balanced_data['steering'], bins=31)
    plt.title("Balanced Steering Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = "collected_data/driving_log.csv"
    img_folder = "collected_data/IMG"

    if not os.path.exists(csv_path):
        print("CSV file not found.")
    else:
        data = load_data(csv_path)
        data = fix_center_paths(data, img_folder)

        balanced_data = balance_data(data)

        show_basic_info("Original Data", data)
        show_basic_info("Balanced Data", balanced_data)

        plot_histograms(data, balanced_data)