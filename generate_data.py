import pandas as pd
import numpy as np

def generate_ctr_data(num_samples=100000, missing_rate=0.05):
    """
    Generates a synthetic dataset for CTR prediction simulating real-world digital advertising data.
    """
    np.random.seed(42)
    
    # 1. User Characteristics
    ages = np.random.randint(18, 65, size=num_samples)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=num_samples, p=[0.48, 0.48, 0.04])
    income_levels = np.random.choice(['Low', 'Medium', 'High'], size=num_samples, p=[0.3, 0.5, 0.2])
    
    # 2. Contextual Features
    devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=num_samples, p=[0.6, 0.35, 0.05])
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=num_samples)
    day_of_week = np.random.randint(0, 7, size=num_samples) # 0=Monday, 6=Sunday
    
    # 3. Ad Characteristics
    ad_categories = np.random.choice(['Electronics', 'Fashion', 'Finance', 'Automotive', 'Entertainment'], size=num_samples)
    ad_placements = np.random.choice(['Sidebar', 'Header', 'Footer', 'In-feed'], size=num_samples)
    
    # 4. Feature Engineering for Synthetic Click Probability
    # We create rules to give the model something to learn
    
    base_ctr = 0.05
    
    # Initialize probabilities
    probs = np.full(num_samples, base_ctr)
    
    # Rule 1: Mobile users have higher CTR for Electronics and Fashion
    mask_mobile_elec = (devices == 'Mobile') & (np.isin(ad_categories, ['Electronics', 'Fashion']))
    probs[mask_mobile_elec] += 0.08
    
    # Rule 2: High income users click more on Finance and Automotive
    mask_high_inc = (income_levels == 'High') & (np.isin(ad_categories, ['Finance', 'Automotive']))
    probs[mask_high_inc] += 0.12
    
    # Rule 3: Evening and Night have slightly lower CTR generally
    mask_night = np.isin(time_of_day, ['Evening', 'Night'])
    probs[mask_night] -= 0.02
    
    # Rule 4: In-feed placement is much better than Footer
    probs[ad_placements == 'In-feed'] += 0.06
    probs[ad_placements == 'Footer'] -= 0.03
    
    # Rule 5: Younger users (18-25) click more on Entertainment
    mask_young_ent = (ages <= 25) & (ad_categories == 'Entertainment')
    probs[mask_young_ent] += 0.10
    
    # Clip probabilities between 0.01 and 0.99
    probs = np.clip(probs, 0.01, 0.99)
    
    # 5. Target Variable Generation
    clicks = np.random.binomial(1, probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_age': ages,
        'user_gender': genders,
        'user_income': income_levels,
        'device_type': devices,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'ad_category': ad_categories,
        'ad_placement': ad_placements,
        'is_click': clicks
    })
    
    # 6. Introduce Missing Values (to satisfy preprocessing requirement)
    for col in ['user_age', 'user_income', 'ad_placement']:
        mask = np.random.rand(num_samples) < missing_rate
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == "__main__":
    print("Generating synthetic CTR dataset...")
    df_ctr = generate_ctr_data(num_samples=100000, missing_rate=0.03)
    
    # Save to CSV
    dataset_path = "ctr_data.csv"
    df_ctr.to_csv(dataset_path, index=False)
    
    print(f"Dataset generated and saved to {dataset_path}")
    print(df_ctr.head())
    print("\nDataset Info:")
    print(df_ctr.info())
    print("\nClick Distribution:")
    print(df_ctr['is_click'].value_counts(normalize=True))
