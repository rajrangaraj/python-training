"""
Demonstration of Flask GraphQL integration using Ariadne.
"""

from flask import Flask, request, jsonify
from ariadne import load_schema_from_path, make_executable_schema, graphql_sync
from ariadne import QueryType, MutationType, SubscriptionType
from ariadne.constants import PLAYGROUND_HTML
from ariadne.asgi import GraphQL
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import json

app = Flask(__name__)

# Data models
@dataclass
class User:
    id: int
    name: str
    email: str
    posts: List['Post']

@dataclass
class Post:
    id: int
    title: str
    content: str
    author_id: int
    comments: List['Comment']

@dataclass
class Comment:
    id: int
    content: str
    post_id: int
    author_id: int

# Mock database
users = {}
posts = {}
comments = {}

# Type definitions
type_defs = """
    type User {
        id: ID!
        name: String!
        email: String!
        posts: [Post!]!
    }

    type Post {
        id: ID!
        title: String!
        content: String!
        author: User!
        comments: [Comment!]!
    }

    type Comment {
        id: ID!
        content: String!
        post: Post!
        author: User!
    }

    type Query {
        user(id: ID!): User
        users: [User!]!
        post(id: ID!): Post
        posts: [Post!]!
        searchPosts(query: String!): [Post!]!
    }

    type Mutation {
        createUser(name: String!, email: String!): User!
        createPost(title: String!, content: String!, authorId: ID!): Post!
        createComment(content: String!, postId: ID!, authorId: ID!): Comment!
        updatePost(id: ID!, title: String, content: String): Post!
        deletePost(id: ID!): Boolean!
    }

    type Subscription {
        postCreated: Post!
        commentAdded(postId: ID!): Comment!
    }
"""

# Resolvers
query = QueryType()
mutation = MutationType()
subscription = SubscriptionType()

@query.field("user")
def resolve_user(*_, id):
    """Resolve user by ID."""
    return users.get(int(id))

@query.field("users")
def resolve_users(*_):
    """Resolve all users."""
    return list(users.values())

@query.field("post")
def resolve_post(*_, id):
    """Resolve post by ID."""
    return posts.get(int(id))

@query.field("posts")
def resolve_posts(*_):
    """Resolve all posts."""
    return list(posts.values())

@query.field("searchPosts")
def resolve_search_posts(*_, query):
    """Search posts by title or content."""
    query = query.lower()
    return [
        post for post in posts.values()
        if query in post.title.lower() or query in post.content.lower()
    ]

@mutation.field("createUser")
def resolve_create_user(*_, name, email):
    """Create a new user."""
    user_id = len(users) + 1
    user = User(id=user_id, name=name, email=email, posts=[])
    users[user_id] = user
    return user

@mutation.field("createPost")
async def resolve_create_post(*_, title, content, authorId):
    """Create a new post."""
    post_id = len(posts) + 1
    author = users.get(int(authorId))
    if not author:
        raise ValueError("Author not found")
    
    post = Post(
        id=post_id,
        title=title,
        content=content,
        author_id=author.id,
        comments=[]
    )
    posts[post_id] = post
    author.posts.append(post)
    
    # Notify subscribers
    await subscription.broadcast(
        "postCreated",
        {"postCreated": post}
    )
    
    return post

@mutation.field("createComment")
async def resolve_create_comment(*_, content, postId, authorId):
    """Create a new comment."""
    comment_id = len(comments) + 1
    post = posts.get(int(postId))
    author = users.get(int(authorId))
    
    if not post or not author:
        raise ValueError("Post or author not found")
    
    comment = Comment(
        id=comment_id,
        content=content,
        post_id=post.id,
        author_id=author.id
    )
    comments[comment_id] = comment
    post.comments.append(comment)
    
    # Notify subscribers
    await subscription.broadcast(
        "commentAdded",
        {"commentAdded": comment, "postId": postId}
    )
    
    return comment

@subscription.source("postCreated")
async def post_created_source(*_):
    """Source for post creation subscription."""
    while True:
        post = await subscription.get("postCreated")
        yield post

@subscription.field("postCreated")
def post_created_resolver(post, *_):
    """Resolve post creation subscription."""
    return post

@subscription.source("commentAdded")
async def comment_added_source(_, info, postId):
    """Source for comment addition subscription."""
    while True:
        event = await subscription.get("commentAdded")
        if event["postId"] == postId:
            yield event["commentAdded"]

@subscription.field("commentAdded")
def comment_added_resolver(comment, *_):
    """Resolve comment addition subscription."""
    return comment

# Create executable schema
schema = make_executable_schema(type_defs, [query, mutation, subscription])

# GraphQL routes
@app.route("/graphql", methods=["GET"])
def graphql_playground():
    """Serve GraphQL Playground."""
    return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    """Handle GraphQL queries."""
    data = request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=app.debug
    )
    status_code = 200 if success else 400
    return jsonify(result), status_code

# Example queries
EXAMPLE_QUERIES = {
    "createUser": """
        mutation {
            createUser(name: "John Doe", email: "john@example.com") {
                id
                name
                email
            }
        }
    """,
    "createPost": """
        mutation {
            createPost(
                title: "Hello GraphQL",
                content: "This is my first post",
                authorId: "1"
            ) {
                id
                title
                content
                author {
                    name
                }
            }
        }
    """,
    "getPosts": """
        query {
            posts {
                id
                title
                content
                author {
                    name
                }
                comments {
                    content
                    author {
                        name
                    }
                }
            }
        }
    """,
    "searchPosts": """
        query {
            searchPosts(query: "GraphQL") {
                id
                title
                content
            }
        }
    """
}

if __name__ == "__main__":
    app.run(debug=True) 