from numpy.random import default_rng
class ClientSelector():
    def random_selector(self, number_of_clients, clients_per_round):
        rng = default_rng()
        return rng.choice(number_of_clients, size=clients_per_round, replace=False)
