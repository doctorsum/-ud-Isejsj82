FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt upgrade -y && \
    apt install -y wget python3-pip && \
    apt install -y g++ make python3 git

# Install Python packages
RUN pip3 install -U "huggingface_hub[cli]" tensorflow numpy python-chess flask

# Download and extract Stockfish
WORKDIR /root
RUN git clone https://github.com/doctorsum/Simple-HTML-PAGE-FOR-UPTIME /root/page
RUN echo "#!""/bin/bash" > passo
RUN printf "expect <<EOF\n" >> passo
RUN printf "spawn huggingface-cli login\n" >> passo
RUN printf 'expect "Enter your token (input will not be visible):"\n' >> passo
RUN printf 'send "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\r"\n' >> passo
RUN printf 'expect "Add token as git credential? (Y/n)"\n' >> passo
RUN printf 'send "y\r"\n' >> passo
RUN printf 'expect eof\n' >> passo
RUN printf 'EOF\n' >> passo
RUN chmod +x passo
RUN ./passo
RUN wget https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx512.tar && \
    tar -xvf stockfish-ubuntu-x86-64-avx512.tar

# Create training script
RUN echo "import chess\n" > main.py && \
    echo "import chess.engine\n" >> main.py && \
    echo "import numpy as np\n" >> main.py && \
    echo "import tensorflow as tf\n" >> main.py && \
    echo "import os, subprocess\n\n" >> main.py && \
    echo "stockfish_path = '/root/stockfish/stockfish-ubuntu-x86-64-avx512'\n" >> main.py && \
    echo "engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)\n" >> main.py && \
    echo "model_file = 'chess_model.h5'\n" >> main.py && \
    echo "elo_file = 'elo.txt'\n" >> main.py && \
    echo "if os.path.exists(model_file):\n" >> main.py && \
    echo "    model = tf.keras.models.load_model(model_file)\n" >> main.py && \
    echo "    print('Model loaded.')\n" >> main.py && \
    echo "else:\n" >> main.py && \
    echo "    model = tf.keras.Sequential([\n" >> main.py && \
    echo "        tf.keras.layers.Dense(128, input_shape=(64,), activation='relu'),\n" >> main.py && \
    echo "        tf.keras.layers.Dense(128, activation='relu'),\n" >> main.py && \
    echo "        tf.keras.layers.Dense(1, activation='sigmoid')\n" >> main.py && \
    echo "    ])\n" >> main.py && \
    echo "    model.compile(optimizer='adam', loss='mse')\n" >> main.py && \
    echo "if os.path.exists(elo_file):\n" >> main.py && \
    echo "    with open(elo_file, 'r') as f:\n" >> main.py && \
    echo "        current_elo = int(f.read())\n" >> main.py && \
    echo "else:\n" >> main.py && \
    echo "    current_elo = 2000\n" >> main.py && \
    echo "def update_elo(player_elo, opponent_elo, result):\n" >> main.py && \
    echo "    K = 32\n" >> main.py && \
    echo "    expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))\n" >> main.py && \
    echo "    return player_elo + K * (result - expected_score)\n" >> main.py && \
    echo "def board_to_input(board):\n" >> main.py && \
    echo "    board_fen = board.board_fen()\n" >> main.py && \
    echo "    board_array = np.array([1 if c == 'P' else -1 if c == 'p' else 0 for c in board_fen if c != '/'])\n" >> main.py && \
    echo "    return np.reshape(board_array, (1, 64))\n" >> main.py && \
    echo "def train_against_stockfish():\n" >> main.py && \
    echo "    global current_elo\n" >> main.py && \
    echo "    stockfish_elo = 3500\n" >> main.py && \
    echo "    board = chess.Board()\n" >> main.py && \
    echo "    while not board.is_game_over():\n" >> main.py && \
    echo "        input_data = board_to_input(board)\n" >> main.py && \
    echo "        predictions = model.predict(input_data)\n" >> main.py && \
    echo "        move = np.argmax(predictions)\n" >> main.py && \
    echo "        legal_moves = list(board.legal_moves)\n" >> main.py && \
    echo "        board.push(legal_moves[move % len(legal_moves)])\n" >> main.py && \
    echo "        result = engine.play(board, chess.engine.Limit(time=1))\n" >> main.py && \
    echo "        board.push(result.move)\n" >> main.py && \
    echo "    game_result = board.result()\n" >> main.py && \
    echo "    result = 1 if game_result == '1-0' else 0 if game_result == '0-1' else 0.5\n" >> main.py && \
    echo "    current_elo = update_elo(current_elo, stockfish_elo, result)\n" >> main.py && \
    echo "    print(f'New Elo: {current_elo}')\n" >> main.py && \
    echo "    with open(elo_file, 'w') as f:\n" >> main.py && \
    echo "        f.write(str(current_elo))\n" >> main.py && \
    echo "    model.save(model_file)\n" >> main.py && \
    echo "    model.fit(input_data, np.array([result]), epochs=250)\n" >> main.py && \
    echo "    upload_to_huggingface()\n" >> main.py && \
    echo "def upload_to_huggingface():\n" >> main.py && \
    echo "    subprocess.run(['huggingface-cli', 'upload', model_file], check=True)\n" >> main.py && \
    echo "    subprocess.run(['huggingface-cli', 'upload', elo_file], check=True)\n" >> main.py && \
    echo "game_number = 1\n" >> main.py && \
    echo "while True:\n" >> main.py && \
    echo "    print(f'Game {game_number}')\n" >> main.py && \
    echo "    train_against_stockfish()\n" >> main.py && \
    echo "    game_number += 1\n"

CMD python3 /root/main.py & python3 /root/page/app.py
