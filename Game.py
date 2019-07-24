from GameBoard import *
from NeuralNetwork import *
import numpy as np
import random

class Game:

    def __init__(self):
        self.board = GameBoard()
        self.nn = NeuralNetwork([15, 200, 6], 0.3)

    def turn(self, player, pit):
        next_player = self.board.move_marbles(player, pit)

        if self.board.check_end_condition():
            # print("end")

            return -1

        return next_player

    def text_play(self):
        player = 0
        print("Welcome to the game of Mancala! Player 0 will go first")
        while player >= 0:
            self.board.text_display_board()
            pit = int(input("Player {}, please enter which pit (1-6) you would like to shift from:".format(player)))

            player = self.turn(player, pit)

        score = self.board.get_score()
        final_marbles = self.board.get_sum_rows()

        score = [score[i] + final_marbles[i] for i in range(len(score))]
        winner = max(enumerate(score), key=lambda x: x[1])[0]

        self.board.text_display_board()
        print("Congratulations Player {}! You are the winner. "
              "The final score was Player 0: {} Player 1: {}".format(winner, score[0], score[1]))
        self.reset_game()

    def reset_game(self):
        self.board = GameBoard()

    @staticmethod
    def random_ai():
        return random.randint(1, 6)

    @staticmethod
    def discount_rewards(reward_history, d_rate):
        discounted_rewards = []

        accumulated_reward = 0
        for i in range(1, len(reward_history) + 1):
            accumulated_reward = reward_history[-i] + accumulated_reward * d_rate
            discounted_rewards.insert(0, accumulated_reward)
            # print(accumulated_reward)

        # normalize rewards before returning
        # print(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards)

        return discounted_rewards

    def train_neural_network(self, game_runs):

        game_input_history = []
        game_target_history = []
        reward_history = []
        player = 0
        reset_counter = 0
        turn_counter = 0
        ai_win_counter = 0
        ai_player = 0

        print("Welcome to the game of Mancala! this training session will continue for {} runs.".format(game_runs))

        while reset_counter < game_runs:
            initial_score = self.board.get_score()
            turn_counter += 1
            if player == ai_player:
                ai_input = self.board.get_board()
                ai_input.append(ai_player)
                game_input_history.append(ai_input)
                ai_decision = self.nn.query(ai_input)

                sorted_decision = list(enumerate(ai_decision.flatten()))
                sorted_decision.sort(key=lambda x: x[1])

                for i in range(len(sorted_decision)):
                    pit = sorted_decision[-(i + 1)][0] + 1
                    next_player = self.turn(player, pit)

                    if next_player == 2:
                        next_player = ai_player
                        #print(self.board.get_score())
                    else:
                        break

                post_turn_score = self.board.get_score()
                score_diff = [post_turn_score[i] - initial_score[i] for i in range(len(initial_score))]

                # Converting the numpy array returned by query into a python list with integer elements (rather than
                # numpy array elements as defaulted to by running list[ai_decision
                ai_decision = [x[0] for x in list(ai_decision)]
                target_list = ai_decision
                # target_list = np.zeros(6)
                if score_diff[player] > 0:
                    target_list[pit - 1] = 0.99
                    reward_history.append(0.5 * score_diff[player])
                else:
                    target_list[pit - 1] = 0.01
                    reward_history.append(-1)

                game_target_history.append(target_list)
                # self.nn.train(ai_input, target_list)

                # print("Turn {} has passed. NN scores {} points this round.".format(turn_counter, score_diff[player]))
            else:
                pit = Game.random_ai()
                next_player = self.turn(player, pit)
                if next_player == 2:
                    next_player = -ai_player + 1

            if next_player == -1:
                score = self.board.get_score()
                final_marbles = self.board.get_sum_rows()

                score = [score[i] + final_marbles[i] for i in range(len(score))]
                winner = max(enumerate(score), key=lambda x: x[1])[0]

                # print("NN: Player {} Random AI: Player {}".format(ai_player, -ai_player + 1))
                # print("Player {} won!".format(winner))

                if winner == ai_player:
                    ai_win_counter += 1
                    reward_history = [reward * 1.5 for reward in reward_history]

                discounted_rewards = Game.discount_rewards(reward_history, 0.2)

                for i in range(len(game_input_history)):
                    self.nn.train(game_input_history[i], [target * discounted_rewards[i]
                                                          for target in game_target_history[i]])

                game_target_history = []
                game_input_history = []
                self.reset_game()
                reset_counter += 1
                turn_counter = 0
                ai_player = -ai_player + 1
                next_player = 0
                if reset_counter > 0 and reset_counter * 100 / game_runs % 10 == 0:
                    print("Training is {}% finished!".format(reset_counter * 100 / game_runs))

            player = next_player

        print("AI has completed this training session.")
        print("AI has won {} percentage of games".format(ai_win_counter/game_runs))

    def test_neural_network(self, game_runs):
        player = 0
        reset_counter = 0
        turn_counter = 0
        ai_win_counter = 0
        ai_player = 0
        next_player = 1

        print("Welcome to the game of Mancala! AI will start first")
        while reset_counter < game_runs:
            initial_score = self.board.get_score()
            turn_counter += 1

            # print(self.board.text_display_board())
            if player == ai_player:
                ai_input = self.board.get_board()
                ai_input.append(ai_player)
                ai_decision = self.nn.query(ai_input)

                sorted_decision = list(enumerate(ai_decision.flatten()))
                sorted_decision.sort(key=lambda x: x[1])

                for i in range(len(sorted_decision)):
                    pit = sorted_decision[-(i + 1)][0] + 1
                    next_player = self.turn(player, pit)

                    if next_player == 2:
                        next_player = ai_player
                        # print(self.board.get_score())
                    else:
                        break

                post_turn_score = self.board.get_score()
                score_diff = [post_turn_score[i] - initial_score[i] for i in range(len(initial_score))]
                # print(post_turn_score)
                print("Turn {} has passed. NN scores {} points this round.".format(turn_counter, score_diff[player]))
            else:
                pit = Game.random_ai()
                next_player = self.turn(player, pit)
                if next_player == 2:
                    next_player = -ai_player + 1

            if next_player == -1:
                score = self.board.get_score()
                final_marbles = self.board.get_sum_rows()

                score = [score[i] + final_marbles[i] for i in range(len(score))]
                winner = max(enumerate(score), key=lambda x: x[1])[0]

                print("NN: Player {} Random AI: Player {}".format(ai_player, -ai_player + 1))
                print("Player {} won!".format(winner))
                print("Final Score is {}!".format(score))
                if winner == ai_player:
                    ai_win_counter += 1
                self.reset_game()
                reset_counter += 1
                turn_counter = 0
                ai_player = -ai_player + 1
                next_player = 0

            player = next_player

        print("AI has won {} percentage of games".format(ai_win_counter / game_runs))

newgame = Game()

#newgame.text_play()
newgame.train_neural_network(10000)
newgame.test_neural_network(1000)
