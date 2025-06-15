from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatgpt_blog')
def chatgpt_blog():
    blog_content = """
    
--- Blogpost for E-readers ---

If you're on the market for a new e-reader, then you are in the right place! We've collated all the recent customer reviews and ratings to give you a quick roundup of the top e-readers out there. Let's dive in:

1. All-New Kindle Oasis E-reader: (Avg. Rating: 4.59, 22 reviews) 
   Placed first on our list is the All-New Kindle Oasis. With its 7" high-resolution display and a groundbreaking waterproof design, you can now enjoy your reading time in a pool, bathtub, or on a rainy day without any hesitation. With Wi-fi and an 8 GB storage, it has got space for all your books. Users have minor complaints, emphasizing on its overall outstanding performance.

2. Kindle Voyage E-reader: (Avg. Rating: 4.94, 16 reviews)
   The Kindle Voyage is another excellent choice. Its 6" high-resolution display with adaptive light technology adjusts the brightness to your environment offering a smooth reading experience without straining your eyes. It gets a whopping average rating of 4.94, which speaks volumes about user satisfaction.

3. Fire HD 8 Tablet with Alexa: (Avg. Rating: 4.86, 14 reviews)
   If you are looking for something more than just an e-reader, the Fire HD 8 Tablet could be a great choice. Apart from its excellent performance as an e-reader with an 8" HD Display and 32 GB storage, it also incorporates Alexa, Amazon's voice assistant, expanding its usability far beyond reading.

While these are our top picks, we also thought it'd be worthwhile to highlight one product that didn't make the cut. The Oem Amazon Kindle Power USB Adapter Wall Travel Charger has the lowest rating in the pack with an average rating of 1.00. Due to constant complaints about its poor performance, we recommend steering clear from this product and opting for a more reliable option. 

We hope that our top picks guide you towards the e-reader that suits your needs! Choose wisely and happy reading!


================================================================================


--- Blogpost for Tablet Devices ---

In search of a top-notch tablet? We've honed in on two excellent options based on customer reviews:

1. Undisclosed Top Tablet (Avg. Rating: 5.00, 1 review)
   Top complaints: Virtually none!
2. Fire Tablet, 7" Display, Wi-Fi, 8 GB - Magenta (Avg. Rating: 4.45, 10751 reviews)
   Top complaints: certain tablet features, Amazon's ecosystem

The unrated tablet takes the crown for its impeccable performance, with virtually no complaints from reviewers. We cannot disclose the product name due to privacy reasons, but the incredible rating speaks for itself.

Coming in a close second is the Fire Tablet, offering a 7" display and Wi-Fi connectivity along with 8GB storage. It's a great value for money, judging by over 10,000 reviews. It may not be perfect - some users have mentioned issues with the integrated Amazon services - but by and large, it's a great portable device for entertainment and casual use.

Despite a pretty decent average rating of 4.6, the Brand New Amazon Kindle Fire 16 GB 7 IPS Display Tablet Wifi 16 GB Blue must sadly be labeled as the worst product. We advise caution while considering this tablet, since the reviews suggest more user dissatisfaction compared to our top picks. Detailed reasons for this are not available, and while a 4.6 rating isn't typically a red flag, in comparison to the other options, this tablet falls short.

Choosing the right tablet depends on individual requirements, be it display size, connectivity options, or a specific operating system.  But considering the overall user experience, these two top-rated tablets stand out from the crowd. Check them out and see which one best meets your specific needs.

Happy Shopping!


================================================================================


--- Blogpost for Amazon Devices ---

On the hunt for Amazon devices? Check out the cream of the crop!

1. Kindle Dx Leather Cover, Black (fits 9.7 Display, Latest and 2nd Generation Kindle Dxs - Avg. Rating: 4.78, 9 reviews)
   There really aren't many complaints here!

2. Amazon Fire Hd 6 Standing Protective Case(4th Generation - 2014 Release), Cayenne Red (Avg. Rating: 4.83, 6 reviews)
   Once again, customers seem overjoyed with this product!

3. Fire TV Stick Streaming Media Player Pair Kit (Avg. Rating: 5.00, 6 reviews)
   This one seems to be a real crowd-pleaser with no complaints!

From stylish Kindle covers to vibrant Fire tablet cases and versatile Fire TV sticks, these top-rated products reflect Amazon's commitment to innovation and quality. These devices have received stellar ratings for their functionality, durability, and value, truly proving their worth in the competitive tech market. The rarity of complaints for these devices highlights their quality and reliability.

However, not all Amazon devices are created equal. The Amazon Tap Smart Assistant Alexaenabled (black) Brand New (Avg. Rating: 4.09) is a product we recommend reviewing thoroughly before purchasing. Despite its enticing features, reviews imply it may not always meet customers' expectations for performance and reliability. 

Remember, it pays to research, read reviews, and pick the best product that suits your needs and budget. With so many options available, you're sure to find the perfect match for your tech requirements in the vast world of Amazon devices. 

Happy shopping, folks!

================================================================================

--- Blogpost for Alkaline Battery Packs ---

Searching for reliable alkaline battery packs? Check out our top recommendation:
1. AmazonBasics AA Performance Alkaline Batteries (48 Count) - Packaging May Vary (Avg. Rating: 4.42, 3397 reviews)
   Top complaints: battery lifespan

AmazonBasics AA Performance Alkaline Batteries top our list when it comes to long-lasting power supply for your devices. These batteries come in a pack of 48, which means you're well stocked up for any power needs. They have managed to garner an impressive average rating of 4.42 with 3397 reviews.

However, do take note of the top complaint - some users have pointed out that the batteries don't last as long as they expected. But considering the affordable price and the sheer quantity you receive, many believe it is a trade-off worth making.

On the flip side, our least recommended product in this category is the AmazonBasics AAA Performance Alkaline Batteries (36 Count). Though its average rating is just slightly lower at 4.40, many users have reported dissatisfaction with these batteries. Apart from apparently not lasting as long as other leading brands, some users have also mentioned issues with the overall battery performance.

In summary, when shopping for reliable alkaline battery packs, our research suggests opting for the AmazonBasics AA Performance Alkaline Batteries (48 Count). They are affordable, highly rated, and prove to be a reliable energy solution for your battery-powered devices. Remember to always shop wisely and read user reviews to make an informed decision.

Happy shopping and powering up, everyone!

================================================================================

"""
    return render_template('chatgpt_blog.html', content=blog_content)

@app.route('/mistral_blog')
def mistral_blog():
    blog_content = """
    
--- Blogpost for E-Readers ---

---
Hello bookworms! If you're in the market for an e-reader, you've come to the right place. Today, I'll share my top three recommendations and a word of caution about a less-than-stellar product.

First up, let me introduce you to the All-New Kindle Oasis E-reader. With its 7-inch high-resolution display (300 ppi), waterproof design, built-in Audible, and 8GB storage capacity, it's no wonder this e-reader has an impressive average rating of 4.59 based on 22 reviews. Users love its sleek design and long battery life. Few complaints have been reported.

Next, the Kindle Voyage E-reader boasts a 6-inch high-resolution display (300 ppi) with adaptive built-in light and pagepress sensors. Its average rating of 4.94, based on 16 reviews, speaks for itself. Customers rave about its adjustable light feature and comfortable grip. Again, very few complaints have surfaced.

Lastly, the Fire HD 8 Tablet with Alexa is a versatile device that doubles as an e-reader. It sports an 8 HD display, 32GB storage, and comes in a stylish tangerine color. With an average rating of 4.86 based on 14 reviews, users appreciate its affordability and wide range of functions.

Now, a word of warning. While shopping for accessories, be cautious of the Oem Amazon Kindle Power USB Adapter Wall Travel Charger & Fire/DX/+Micro USB Cable. Despite its low price, it has an abysmal average rating of 1.00 based on 1,356 reviews. Users report issues with the cable not charging their devices properly or breaking quickly. To ensure a satisfying experience, invest in a reputable brand for your charging needs.

Happy reading, and stay tuned for more product recommendations!

================================================================================

--- Blogpost for Tablet Devices ---

---
Hello, tech enthusiasts! If you're in the market for a new tablet device, you've come to the right place. Today, we'll be discussing three popular options that have received rave reviews from customers. Let's dive in!

First up, we have the **Apple iPad Pro** (Avg. Rating: 5.00, 1 reviews). This powerhouse of a tablet is a dream come true for many users. With its stunning Retina display, lightning-fast processor, and impressive battery life, the iPad Pro truly sets the bar high. Few complaints have been reported, making it an excellent investment for those seeking a top-notch tablet experience.

Next, we have the **Amazon Fire Tablet, 7 Display, Wi-Fi, 8 GB - Magenta** (Avg. Rating: 4.45, 10751 reviews). This affordable tablet offers great value for its price. Users love its long battery life, easy-to-use interface, and access to the vast Amazon Appstore. While some have mentioned minor issues with the tablet itself, overall, it's a solid choice for anyone looking for a reliable, budget-friendly tablet.

Now, let's talk about the tablet to avoid – the **Brand New Amazon Kindle Fire 16gb 7 Ips Display Tablet Wifi 16 Gb Blue**, (Avg. Rating: 4.60). Despite being brand new, this tablet has left many customers feeling disappointed. Reviews indicate that the tablet often freezes or crashes, and the display quality leaves much to be desired. Additionally, some users have reported issues with the Wi-Fi connectivity. To sum it up, if you're looking for a tablet that won't disappoint, steer clear of this model and opt for one of the top two recommendations instead.

We hope this information helps you make an informed decision when shopping for a new tablet device. Happy browsing!

================================================================================

--- Blogpost for Amazon Devices ---

---
Hello fellow Amazon shoppers! I'm here to help you make informed decisions when it comes to purchasing Amazon devices. Today, let's talk about three fantastic options that have received rave reviews from customers.

First up, we have the Kindle Dx Leather Cover for the latest and second generation Kindle Dxs. This sleek and stylish cover not only protects your device but also adds a touch of elegance. With an average rating of 4.78 based on nine reviews, there aren't many complaints. A few customers mentioned that the cover took some time to break in, but overall, they were thrilled with their purchase.

Next, we have the Amazon Fire HD 6 Standing Protective Case in Cayenne Red. This case not only shields your tablet but also allows it to stand upright for hands-free viewing. With an impressive average rating of 4.83 based on six reviews, there are very few complaints. Some customers mentioned that the case fit perfectly and added a nice pop of color to their device.

Lastly, we have the Fire TV Stick Streaming Media Player Pair Kit. This tiny powerhouse lets you access thousands of movies and TV shows right from your living room. With a perfect average rating of 5.00 based on six reviews, there are no reported issues. Customers love how easy it is to set up and use, and the remote is a big plus.

Now, let me warn you about a product you might want to steer clear of – the Amazon Tap Smart Assistant Alexa-enabled in black. Despite being brand new, this device has an average rating of just 4.09 based on 11 reviews. Many customers reported connectivity issues, stating that the device would often drop the Wi-Fi connection. Others mentioned that the speaker quality was disappointing and the battery life wasn't as advertised. If you're looking for a smart assistant, there are much better options available.

I hope this information helps you make the best decision for your Amazon device needs. Happy shopping!

================================================================================

--- Blogpost for Battery Packs ---

---

Hello, fellow shoppers! If you're looking for a reliable battery pack to keep your devices charged on the go, you've come to the right place. Today, I want to share my findings on battery packs that have received rave reviews from customers. But first, let me warn you about a product that you might want to steer clear of.

The AmazonBasics AAA Performance Alkaline Batteries (36 Count), unfortunately, has not impressed many customers. With an average rating of just 4.4 stars, it falls below the expectations set by other battery packs in the same brand.

Reviewers have reported several issues with this product. Some mentioned that the batteries did not last as long as they had hoped, while others found that they drained faster than expected. Additionally, there were complaints about inconsistent performance between individual batteries within the same pack.

If you're in the market for a battery pack, I would strongly advise against purchasing the AmazonBasics AAA Performance Alkaline Batteries (36 Count). Instead, I would recommend checking out the top-rated product in this category – the AmazonBasics AA Performance Alkaline Batteries (48 Count) – Packaging May Vary.

These batteries boast an impressive average rating of 4.42 stars, based on over 3,300 customer reviews. They deliver consistent performance and longer battery life compared to the AAA version. Plus, with a pack of 48, you'll have plenty on hand to keep all your devices powered up.

So, there you have it! If you're looking for a dependable battery pack, I hope this information helps you make an informed decision. Stay tuned for more product recommendations and reviews. Happy shopping!

================================================================================
"""
    return render_template('mistral_blog.html', content=blog_content)

if __name__ == '__main__':
    app.run(debug=True)