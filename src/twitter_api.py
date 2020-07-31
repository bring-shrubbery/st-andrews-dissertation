import tweepy

def tweet(text, image_url=None):
    with open('../twitter-api', 'r') as f:
        api_key = f.readline().strip()
        api_key_secret = f.readline().strip()
        access_token = f.readline().strip()
        access_token_secret = f.readline().strip()
        bearer_token = f.readline().strip()

        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)

        # Create API object
        api = tweepy.API(auth)

        # Create a tweet
        if image_url != None:
            media = api.media_upload(image_url)

            api.update_status(text, media_ids=[media.media_id])
        else:
            api.update_status(text)
