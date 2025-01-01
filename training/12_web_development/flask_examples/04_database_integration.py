"""
Demonstration of Flask with SQLAlchemy database integration.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class User(db.Model):
    """User model."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    posts = db.relationship('Post', backref='author', lazy=True)
    
    def set_password(self, password):
        """Set hashed password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    """Blog post model."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tags = db.relationship('Tag', secondary='post_tags', backref='posts')
    
    def __repr__(self):
        return f'<Post {self.title}>'

class Tag(db.Model):
    """Tag model."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    
    def __repr__(self):
        return f'<Tag {self.name}>'

# Association table for many-to-many relationship
post_tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

# Routes
@app.route('/')
def index():
    """Show latest blog posts."""
    posts = Post.query.order_by(Post.created_at.desc()).limit(10).all()
    return render_template('blog/index.html', posts=posts)

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    """Show post detail."""
    post = Post.query.get_or_404(post_id)
    return render_template('blog/post_detail.html', post=post)

@app.route('/post/new', methods=['GET', 'POST'])
def new_post():
    """Create new blog post."""
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        tag_names = request.form['tags'].split(',')
        
        # Create or get tags
        tags = []
        for name in tag_names:
            name = name.strip()
            if name:
                tag = Tag.query.filter_by(name=name).first()
                if not tag:
                    tag = Tag(name=name)
                tags.append(tag)
        
        # Create post
        post = Post(
            title=title,
            content=content,
            user_id=1,  # Hardcoded for demo
            tags=tags
        )
        
        try:
            db.session.add(post)
            db.session.commit()
            flash('Post created successfully!', 'success')
            return redirect(url_for('post_detail', post_id=post.id))
        except Exception as e:
            db.session.rollback()
            flash('Error creating post. Please try again.', 'error')
    
    return render_template('blog/new_post.html')

@app.route('/tag/<tag_name>')
def tag_posts(tag_name):
    """Show posts with specific tag."""
    tag = Tag.query.filter_by(name=tag_name).first_or_404()
    return render_template('blog/tag_posts.html', tag=tag)

@app.route('/user/<username>')
def user_posts(username):
    """Show user's posts."""
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(user_id=user.id).order_by(Post.created_at.desc()).all()
    return render_template('blog/user_posts.html', user=user, posts=posts)

# Database commands
@app.cli.command('init-db')
def init_db():
    """Initialize the database."""
    db.create_all()
    print('Database initialized.')

@app.cli.command('seed-db')
def seed_db():
    """Seed the database with sample data."""
    # Create a user
    user = User(username='demo_user', email='demo@example.com')
    user.set_password('password123')
    db.session.add(user)
    
    # Create some tags
    tags = [
        Tag(name='python'),
        Tag(name='flask'),
        Tag(name='web')
    ]
    db.session.add_all(tags)
    
    # Create some posts
    posts = [
        Post(
            title='Getting Started with Flask',
            content='Flask is a lightweight WSGI web application framework...',
            user_id=1,
            tags=tags[:2]
        ),
        Post(
            title='SQLAlchemy Basics',
            content='SQLAlchemy is the Python SQL toolkit and ORM...',
            user_id=1,
            tags=[tags[0]]
        )
    ]
    db.session.add_all(posts)
    
    try:
        db.session.commit()
        print('Database seeded.')
    except Exception as e:
        db.session.rollback()
        print(f'Error seeding database: {e}')

if __name__ == '__main__':
    app.run(debug=True) 