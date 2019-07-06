
class GameBoard:

    def __init__(self):
        self.board = [0,
                      4, 4, 4, 4, 4, 4,
                      4, 4, 4, 4, 4, 4,
                      0]

    def move_marbles(self, player, pit):
        assert player == 0 or player == 1, "Invalid player id, must be either 0 or 1"
        assert 1 <= pit <= 6, "Invalid pit number {}, must be between 1 and 6".format(pit)

        index = GameBoard.board_index(player, pit)
        marbles = self.board[index]

        if marbles == 0:
            return 2

        self.board[index] = 0

        while marbles != 0:
            index = GameBoard.next_index(index, player)
            # print(index)
            self.board[index] += 1
            marbles -= 1

        if index == 0 or index == 13:
            return player

        if self.board[index] == 1:
            across = GameBoard.index_across(index)
            if self.board[across] != 0:
                plus_points = self.board[across]
                plus_points += self.board[index]
                self.board[across] = 0
                self.board[index] = 0
                self.board[-player] += plus_points

        return -player + 1

    def check_end_condition(self):
        index0 = GameBoard.board_index(0, 1)
        index1 = GameBoard.board_index(1, 1)
        return not any(self.board[index0: index0 + 6]) or not any(self.board[index1: index1 + 6])

    def get_sum_rows(self):
        index0 = GameBoard.board_index(0, 1)
        index1 = GameBoard.board_index(1, 1)
        return sum(self.board[index0: index0 + 6]), sum(self.board[index1: index1 + 6])

    def get_score(self):
        return self.board[0], self.board[-1]

    def get_board(self):
        return self.board[:]

    def text_display_board(self):
        separator = "========================="
        print(separator)

        spacing = "  "
        p0_row = spacing * 2
        for i in range(1, 7):
            p0_row += str(self.board[i])
            p0_row += spacing
        print(p0_row)

        score_row = spacing
        for i in range(14):
            if i == 0 or i == 13:
                score_row += str(self.board[i])
            elif i < 7:
                score_row += spacing + " "
        print(score_row)

        p1_row = spacing * 2
        for i in range(7, 13):
            p1_row += str(self.board[i])
            p1_row += spacing
        print(p1_row)

        print(separator)

    @staticmethod
    def index_across(index):
        if index > 6:
            return index - 6
        if index <= 6:
            return index + 6

    @staticmethod
    def board_index(player, pit):
        """
        :param player: 0 or 1
        :param pit: numbered 1 - 6
        :return: returns index on the board for the pit depending on player, calculated using player * 6 + pit
        """
        return player * 6 + pit

    @staticmethod
    def next_index(index, player):
        if index == 0:
            return 7
        if index == 13:
            return 6

        if index <= 6:
            index = index - 1
            if index == 0 and player == 1:
                return 7
            return index
        if index <= 12:
            index = index + 1
            if index == 13 and player == 0:
                return 6
            return index
