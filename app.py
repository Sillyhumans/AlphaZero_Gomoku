# server.py
from flask import Flask, request, jsonify, render_template
import pickle
from game import Board, Game
import numpy as np
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 初始化游戏参数
n = 5
width, height = 8, 8
model_file = './model/best_policy_8_8_5.model'
policy_param = pickle.load(open(model_file, 'rb'), encoding='latin1')
best_policy = PolicyValueNetNumpy(width, height, policy_param)
mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

# 创建棋盘
board = Board(width=width, height=height, n_in_row=n)
board.init_board(start_player=0)
game = Game(board)

@app.route('/reset', methods=['POST'])
def reset():
    global board, mcts_player
    board.init_board(start_player=0)   # 清空棋盘
    mcts_player.reset_player()         # 如果 MCTS 实现有 reset 接口，记得一起清
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    print(data)
    human_move = data.get("move")  # e.g., [2,3]
    if human_move is None or not isinstance(human_move, list):
        return jsonify({'error': 'Invalid move format'}), 400

    move = board.location_to_move(human_move)
    if move not in board.availables:
        return jsonify({'error': 'Invalid move (occupied or out of range)'}), 400

    board.do_move(move)
    if board.game_end()[0]:
        return jsonify({'winner': board.game_end()[1], 'move': human_move, 'gameover': True})

    ai_move = mcts_player.get_action(board)
    board.do_move(ai_move)
    ai_location = board.move_to_location(ai_move)

    end, winner = board.game_end()

    return jsonify({
            'human_move': [int(x) for x in human_move],
            'ai_move': [int(x) for x in ai_location],
            'winner': int(winner) if end and winner is not None else None,
            'gameover': end
    })

if __name__ == '__main__':
    app.run(debug=True)
