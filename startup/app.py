from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import csv
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import io
import pickle 
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATABASE = 'startups.db'

with open('model/nlogreg_model.pkl', 'rb') as file: 
    model = pickle.load(file)

# Database functions
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS startup (
            id TEXT PRIMARY KEY,
            umur_startup INTEGER,
            jumlah_investor INTEGER,
            nilai_investasi INTEGER,
            ukuran_tim INTEGER,
            industri TEXT,
            berhasil TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Routes

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/pendaftaran', methods=['GET', 'POST'])
def pendaftaran():
    if request.method == 'POST':
        # Get form data
        startup_id = request.form['startup_id']
        umur_startup = request.form['umur_startup']
        jumlah_investor = request.form['jumlah_investor']
        nilai_investasi = request.form['nilai_investasi']
        ukuran_tim = request.form['ukuran_tim']
        industri = request.form['industri']
        berhasil = "TIDAK"  # Default value

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO startup (id, umur_startup, jumlah_investor, nilai_investasi, ukuran_tim, industri, berhasil)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (startup_id, umur_startup, jumlah_investor, nilai_investasi, ukuran_tim, industri, berhasil))
        conn.commit()
        conn.close()

        flash('Startup added successfully', 'success')
        return redirect(url_for('analisis'))

    return render_template('pendaftaran.html')

@app.route('/kriteria')
def kriteria():
    # Fetch from database
    conn = get_db_connection()
    startups = conn.execute('SELECT * FROM startup').fetchall()
    conn.close()

    # Fetch from CSV file, skip the first row (header)
    csv_startups = []
    with open('uploaded.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        csv_startups = list(reader)

    return render_template('kriteria.html', startups=startups, csv_startups=csv_startups)


@app.route('/kriteria/edit/<id>', methods=['GET', 'POST'])
def edit_kriteria(id):
    conn = get_db_connection()
    
    if request.method == 'POST':
        umur_startup = request.form['umur_startup']
        jumlah_investor = request.form['jumlah_investor']
        nilai_investasi = request.form['nilai_investasi']
        ukuran_tim = request.form['ukuran_tim']
        industri = request.form['industri']
        berhasil = request.form['berhasil']
        
        conn.execute('''
            UPDATE startup 
            SET umur_startup = ?, jumlah_investor = ?, nilai_investasi = ?, ukuran_tim = ?, industri = ?, berhasil = ?
            WHERE id = ?
        ''', (umur_startup, jumlah_investor, nilai_investasi, ukuran_tim, industri, berhasil, id))
        conn.commit()
        flash('Startup updated successfully', 'success')
        return redirect(url_for('kriteria'))
    
    startup = conn.execute('SELECT * FROM startup WHERE id = ?', (id,)).fetchone()
    conn.close()
    
    if startup is None:
        flash('Startup not found', 'error')
        return redirect(url_for('kriteria'))
    
    return render_template('edit_kriteria.html', startup=startup)

@app.route('/kriteria/delete/<id>', methods=['POST'])
def delete_kriteria(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM startup WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('Startup deleted successfully', 'success')
    return redirect(url_for('kriteria'))

@app.route('/export')
def export_data():
    conn = get_db_connection()
    startups = conn.execute('SELECT * FROM startup').fetchall()
    conn.close()

    with open('startups_export.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Umur Startup', 'Jumlah Investor', 'Nilai Investasi', 'Ukuran Tim', 'Industri', 'Berhasil'])
        for startup in startups:
            writer.writerow(startup)

    return send_file('startups_export.csv', as_attachment=True)


@app.route('/analisis', methods=['GET'])
def analisis():
    conn = get_db_connection()
    
    # Fetch data from the database
    data = pd.read_sql_query("SELECT * FROM startup", conn)

    feature_cols = ['umur_startup', 'jumlah_investor', 'nilai_investasi', 'ukuran_tim']
    
    X = data[feature_cols]
    y = data['berhasil'].apply(lambda x: 1 if x == 'YA' else 0)

    X.fillna(0, inplace=True)

    # Load the model
    with open('model/nlogreg_model.pkl', 'rb') as file: 
        model = pickle.load(file)

    # Fit the model with current data
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Update 'berhasil' status based on prediction probability
    for index, prob in enumerate(probabilities):
        if prob > 0.5:
            startup_id = data.iloc[index]['id']
            conn.execute('UPDATE startup SET berhasil = ? WHERE id = ?', ('YA', startup_id))
    
    conn.commit()

    # Fetch updated data
    data = pd.read_sql_query("SELECT * FROM startup", conn)
    conn.close()

    results = data[feature_cols + ['berhasil']].copy()
    results['Prediction'] = ['YA' if prob > 0.8 else 'TIDAK' for prob in probabilities]
    results['Success Probability'] = probabilities

    success_counts = data['berhasil'].value_counts()
    labels = ['YA', 'TIDAK']
    values = [success_counts.get('YA', 0), success_counts.get('TIDAK', 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['green', 'red'])
    plt.title('Startup Success vs Failure')
    plt.xlabel('Outcome')
    plt.ylabel('Number of Startups')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    success_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('analisis.html', 
                           data=results.to_dict(orient='records'), 
                           total_startups=len(data),
                           model_name='nlogreg_model',
                           success_plot_url=success_plot_url)



def load_startups_data():
    df = pd.read_csv('data/startups.csv')

    # Ensure the data is in the correct type
    df['umur_startup'] = pd.to_numeric(df['umur_startup'], errors='coerce')
    df['jumlah_investor'] = pd.to_numeric(df['jumlah_investor'], errors='coerce')
    df['nilai_investasi'] = pd.to_numeric(df['nilai_investasi'], errors='coerce')
    df['ukuran_tim'] = pd.to_numeric(df['ukuran_tim'], errors='coerce')

    return df.to_dict(orient='records')

# Initialize database on first run
if __name__ == '__main__':
    init_db()
    app.run(debug=True)

