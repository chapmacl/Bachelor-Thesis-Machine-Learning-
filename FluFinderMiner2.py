import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import string
import json

# consumer key, consumer secret, access token, access secret.
ckey = "ZD26bhJnJaFqCkwcsSxTeRmT7"
csecret = "C92pAnJdQKnPse1EZtdhsmVm4yNCS2tJw7MMHgZUqdYBoo0sdC"
atoken = "940613974067957762-dpqASwm9vKDqqpypCPKKDaMU5y5LyA4"
asecret = "8RXhDyS4SeAqFzoLBARJMXFDzGdnpm0i8DJC55lvGSSRX"


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def on_data(self, data):
        
        self.outfile = "flu_tweets.json" 
        try:
            with open(self.outfile, 'a') as f:
                date = "date: " + data.split('"created_at":"')[1].split('","id"')[0]
                tweet = "tweet: " + data.split(',"text":"')[1].split(',"source')[0]
                location = "location: " + data.split(',"location":"')[1].split('","url"')[0]
                """
                decoded = json.loads(data);
                tweet = decoded['text'];
                screen_name = decoded['user']['screen_name']
                """
                out = date + ", " + tweet + ", " + ", " + ", " + location + "\n\n"
                # f.write(out)
                print(out)
                
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert file name into a safe string.
    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.
    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


if __name__ == '__main__':
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    api = tweepy.API(auth)

    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(track=["flu "])
