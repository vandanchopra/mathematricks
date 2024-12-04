from dotenv import load_dotenv
import os, time, requests

load_dotenv()
        
class TelegramBot:
    def __init__(self):
        """
        Initialize the bot with the provided Telegram bot token.
        
        :param token: The API token for the Telegram bot
        """
        self.token = os.getenv('MATHEMATRICKS_TELEGRAM_BOT_TOKEN')
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        # self.bot_chat_id = '781528652'
        self.mathematricks_group_chat_id = '-365852442'

    def send_message(self, message):
        """
        Sends a message to a specific chat ID.
        
        :param chat_id: The Telegram chat ID where the message will be sent
        :param message: The message to send
        :return: The response from the Telegram API
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.mathematricks_group_chat_id,
            'text': message
        }
        response = requests.post(url, data=payload)
        time.sleep(0.5)
        
        if response.status_code == 200:
            return response.json()  # Message successfully sent
        else:
            raise Exception(f"Error sending message: {response.text}")

# Example usage:
if __name__ == "__main__":
      # Replace with the private group chat ID
    message = 'NEW MESSAGE FROM MATHEMATRICKS'

    bot = TelegramBot()
    response = bot.send_message(message)
    print(response)