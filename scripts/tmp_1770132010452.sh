NEED_CONFIRM
python3 -c '
class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            return self.balance
        else:
            return "Insufficient funds"

account = BankAccount(100)
print("Initial balance:", account.deposit(50))
print("Withdraw 20:", account.withdraw(20))
print("Deposit 30:", account.deposit(30))
print("Final balance:", account.withdraw(100))
'