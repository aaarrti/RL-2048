### A RL agent learning to play 2048

### Run Game ENV
    go get .
    go run 2048/main.go

### Run agent
    cd agent
    docker build --tag rnn .
    docker run rnn
    