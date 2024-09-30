import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

@app.route('/analisis_startup')
def analisis_startup():
    # Load data from CSV (replace 'data/startups.csv' with the actual path)
    df = pd.read_csv('data/startups.csv')

    # Example DataFrame columns: ['id', 'umur_startup', 'jumlah_investor', 'nilai_investasi', 'ukuran_tim', 'industri', 'berhasil']

    # Plot 1: Distribution of Startups by Industry
    plt.figure(figsize=(10, 6))
    sns.countplot(x='industri', data=df)
    plt.title('Distribution of Startups by Industry')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.xlabel('Industry')
    plt.ylabel('Number of Startups')

    # Save plot to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    industry_plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Plot 2: Success Rate by Industry
    plt.figure(figsize=(10, 6))
    sns.countplot(x='industri', hue='berhasil', data=df)
    plt.title('Success Rate by Industry')
    plt.xticks(rotation=45)
    plt.xlabel('Industry')
    plt.ylabel('Count of Startups')

    # Save second plot to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    success_plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Render template, pass plot URLs and the startups data to template
    return render_template('analisis_startup.html', industry_plot_url=industry_plot_url, success_plot_url=success_plot_url, startups=df.to_dict(orient='records'))
