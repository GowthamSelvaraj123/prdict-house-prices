<h1>Boston House Price Prediction</h1>

<p>This project predicts housing prices in Boston using <strong>multivariable linear regression</strong>. It includes data exploration, scaling, regression modeling, residual analysis, and prediction of new house prices.</p>
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/49d86929-73f3-49d2-a139-284b12e3db9b" />
<h2>Features</h2>
<ul>
  <li>Load Boston housing dataset locally (offline-friendly)</li>
  <li>Explore data with summary statistics and correlation heatmaps</li>
  <li>Scale features and split data into training and testing sets</li>
  <li>Fit a multivariable linear regression model</li>
  <li>Evaluate model using RMSE and R² score</li>
  <li>Analyze residuals to check model assumptions</li>
  <li>Use Power Transformation to improve model performance</li>
  <li>Predict the price of a new property based on features</li>
</ul>

<h2>Installation</h2>
<p>Clone this repository and install the required Python libraries:</p>

<pre><code>git clone https://github.com/YOUR_USERNAME/Boston-House-Price-Prediction.git
cd Boston-House-Price-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn
</code></pre>

<h2>Usage</h2>
<p>Run the Python script to see the full pipeline in action:</p>

<pre><code>python app.py
</code></pre>

<h2>Example Output</h2>
<ul>
  <li>RMSE: 4.52</li>
  <li>R² Score: 0.74</li>
  <li>Estimated house price for a new property: $345,000.00</li>
</ul>

<h2>Project Structure</h2>
<ul>
  <li><code>app.py</code> – Main Python script containing the full model pipeline</li>
  <li><code>boston.csv</code> – Boston housing dataset saved locally</li>
  <li><code>README.html</code> – Project documentation</li>
</ul>

<h2>References</h2>
<ul>
  <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html">scikit-learn fetch_openml</a></li>
  <li><a href="https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">Boston Housing Dataset Details</a></li>
</ul>
