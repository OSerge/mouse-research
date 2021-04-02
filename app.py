#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import (
    UserMixin, LoginManager, login_user, logout_user, login_required,
    current_user
)
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.config['SECRET_KEY'] = 'c1ddff3d632e194b36a6c186809372f17db4ff3e922e2342'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)
login_manager.login_message = u'Пожалуйста, введите учетные данные.'

socketio = SocketIO(app, async_mode='eventlet')


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000), unique=True)
    session_num = db.Column(db.Integer, default=0)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
@login_required
def index():
    return render_template("index.html",
                           user_id=current_user.id,
                           session_num=current_user.session_num)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login_post():
    login = request.form.get('login')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    # Проверяем совпадает ли введенный логин с почтой или именем (логином)
    user = User.query.filter(
        (User.email == login) | (User.name == login)).first()

    if not user or not check_password_hash(user.password, password):
        flash('Проверьте введенные данные.')
        return redirect(url_for('login'))

    login_user(user, remember=remember)
    # Увеличиваем счетчик сессий пользователя
    user.session_num += 1
    db.session.commit()
    return redirect(url_for('index'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET'])
def signup():
    return render_template("register.html")


@app.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    user = User.query.filter_by(email=email).first()
    if user:
        flash("Пользователь с такой почтой уже зарегистрирован.")
        return redirect(url_for('signup'))

    new_user = User(
        email=email,
        name=name,
        password=generate_password_hash(password, method='sha256')
    )

    db.session.add(new_user)
    db.session.commit()

    if not os.path.exists(f'./sessions/user{new_user.id}'):
        os.mkdir(f'./sessions/user{new_user.id}')

    return redirect(url_for('login'))


@socketio.on('connection')
def connect():
    print("Client connected.")
    emit("msg", {"msg": "Connected to server.", "type": "info"})


@socketio.on('push')
def get_parameters(message):
    message = json.loads(message)
    print(f"Received message: {message['data']}")

    with open(f"./sessions/user{message['user_id']}"
              f"/session_{message['session_num']}.csv", "a") as file:
        file.write(message['data'])


if __name__ == '__main__':
    socketio.run(app, debug=True)
