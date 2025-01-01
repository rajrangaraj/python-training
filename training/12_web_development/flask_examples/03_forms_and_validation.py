"""
Demonstration of Flask forms, validation, and file uploads.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, TextAreaField, DecimalField, SelectField
from wtforms.validators import DataRequired, Email, Length, NumberRange
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for CSRF protection

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Form classes
class ContactForm(FlaskForm):
    """Contact form with validation."""
    name = StringField('Name', validators=[
        DataRequired(),
        Length(min=2, max=50)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    subject = StringField('Subject', validators=[
        DataRequired(),
        Length(min=5, max=100)
    ])
    message = TextAreaField('Message', validators=[
        DataRequired(),
        Length(min=10, max=1000)
    ])

class ProductForm(FlaskForm):
    """Product submission form with file upload."""
    name = StringField('Product Name', validators=[
        DataRequired(),
        Length(min=2, max=50)
    ])
    description = TextAreaField('Description', validators=[
        DataRequired(),
        Length(min=10, max=500)
    ])
    price = DecimalField('Price', validators=[
        DataRequired(),
        NumberRange(min=0.01)
    ])
    category = SelectField('Category', choices=[
        ('electronics', 'Electronics'),
        ('clothing', 'Clothing'),
        ('books', 'Books'),
        ('other', 'Other')
    ])
    image = FileField('Product Image', validators=[
        FileAllowed(ALLOWED_EXTENSIONS, 'Images only!')
    ])

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Handle contact form."""
    form = ContactForm()
    
    if form.validate_on_submit():
        # Process the form data (e.g., send email)
        flash('Thank you for your message! We will respond shortly.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html', form=form)

@app.route('/submit-product', methods=['GET', 'POST'])
def submit_product():
    """Handle product submission with file upload."""
    form = ProductForm()
    
    if form.validate_on_submit():
        # Handle file upload
        if form.image.data:
            filename = secure_filename(form.image.data.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            form.image.data.save(filepath)
        
        # Process other form data
        product_data = {
            'name': form.name.data,
            'description': form.description.data,
            'price': float(form.price.data),
            'category': form.category.data,
            'image': filename if form.image.data else None
        }
        
        flash('Product submitted successfully!', 'success')
        return redirect(url_for('submit_product'))
    
    return render_template('submit_product.html', form=form)

# Custom error pages
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return render_template('error.html', 
        error='File is too large. Maximum size is 16MB.'), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request error."""
    return render_template('error.html',
        error='Bad request. Please check your form data.'), 400

if __name__ == '__main__':
    app.run(debug=True) 