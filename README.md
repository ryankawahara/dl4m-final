# A Comparison of Visual and Audio Genre Classification in Movie Trailers

Goal: Evaluate the performances of visual and audio models on genre classification for movie trailers.

## Data

dataset.txt is a collection of movies, trailers, and genres from www.themoviedb.org. Almost all of the links provided in the dataset were valid, and we were able to download 93% of them. We downloaded 31 frames as well as the audio from each trailer. The frames were selected at consistent intervals, varying on the length of the trailer. To avoid the green trailer slide, the first frame was always 200 frames in. We also used cv2 to decide to skip any completely black frames. The data had multiple labels, between 1 and 6 in the following genres: 
Action, Adventure, Comedy, Crime, Drama, Horror, Mystery, Romance, Science Fiction, Thriller

## Visual Approach

To reduce the amount of data, we narrowed down the frames selected from each movie to the middle 10. Each frame was evaluated independently of the other. We used VGG16 and Imagenet to extract features from each frame and then ran a Convolutional Neural Network on the features. We chose this method because we thought that using pre-trained models would improve performance. Surprisingly, the model we made from a convolutional neural network to train as a baseline performed slightly better. However, in the demo we used the results from the transfer learning model because we had intended to test the strength of that approach for this task. We thought that the model would perform well in categories like horror, action, or science-fiction due to the amount of visual cues in each genre.

The results were less than great. Accuracy plateaued at about 30% with 100 training epochs and decreased with more. In the predictions, most of the values were between 0.2 and 0.4, so we set a 0.36 decision threshold. The transfer learning model guessed that almost every film was action, adventure, comedy, crime, or drama. Like the audio model, it had a hard time with romance and mystery. The predictions varied most in the categories of crime, horror, and science fiction. Like the audio model, the visual model performed best with drama, with 0.48 precision. However, with a high recall as well, it’s likely it was just guessing that a lot.

The low performance of this frame-based approach is likely because it doesn’t include context. Images taken in isolation can be hard for even a human to guess. Given that the genre “thriller” was often associated with “action,” “drama,” “crime,” and “horror,” dropping that label and simplifying the problem would have been wise. We ran a model where one label of the 10 was picked at random, but did not see any improvement.


## Audio

Based on our knowledge and experience of watching movie trailers, we thought that it would be worthwhile to investigate whether trailer audio was predictive of genre.  If you think about a classic trailer from a given genre, you likely have sounds that you associate with those genres - for example - screams for a horror movie, explosions for an action movie, or laughing for a comedy movie.

We decided that we would limit the data we used in the models to the first 30 seconds of audio from each trailer.  We felt that this time is when trailers try to establish tone and mood, so that these portions of the trailers should carry the most information about genre.

We used 3 models - a relatively basic CNN, and two pre-trained models: YAMNET and VGGISH.

The basic CNN was chosen for two main reasons: (1) it gives a good performance baseline and (2) we were interested in looking at how accurate a model trained on audio windows would be.  Performance, understandably, was pretty bad.

YAMNET and VGGISH were two models that we had higher hopes for - for a couple reasons.

These models look at the trailer audio in its entirety, rather than looking at a specific window.  This is really important in this context, because trailers are a collection of discrete moments from the movie rather than a continuous story.  That means different parts of the trailer can have drastically different sounds/events/moods associated with them.  Because these models look at the full context of the audio, we felt they would perform better.

These models also are trained on audio event data.  To us, trailers can be thought of as a series of somewhat discrete events, so we felt that these models would be able to pick up on different events happening in the trailers, and classify each combination of events as belonging to one or several genres.

The pre-trained models perform significantly better than the base CNN: ~40% test accuracy compared to 29%.  We were able to overfit the models (even with relatively small hyperparameters training accuracy can go up to about 55-60%) which implies to us that there is more room for improvement here.

Looking at prediction patterns for each class and each model don’t provide many clear insights, but we did see a couple interesting patterns.
 
Drama is a clear standout: the pre-trained models have relatively good precision (49% and 46% at a 50% decision threshold, 49% and 48% at a 10% decision threshold, respectively) for YAMNET and VGGISH.  This does not match our intuition at all, we thought Drama would be one of, if not the single hardest genre to predict.  It doesn’t seem as prone to the stereotypes seen in other genres.  

The models had no idea on Mystery, Romance, and Sci-Fi.  Mystery and Romance make sense to an extent - these are broadly subgenres that can contain heavy elements of other genres in our dataset.  Mystery movies are often also Action, Adventure, Drama, or Thriller movies, and Romance movies are often Comedies or Dramas.  Sci-Fi is an interesting one.  Sci-Fi on first thought can also be several other genres, but what really separates Sci-Fi from other genres is the imagery - audio is often closer to the “secondary” genre of that move (Action, Horror, etc.).

Another interesting point: at a 10% decision threshold, all 3 models exhibit much improved performance.  Recall obviously increases significantly, but precision stays steady or increases across models (11%, 26%, 28% to 28%, 28%, 28%).  As you lower the threshold, it causes the model to guess more often that a given sample belongs to a given class.  But those increased guesses look to be right at an above-average rate (relative to the model’s performance).  The more guesses the models make, the better their precision is, meaning the “confidence level” for many correct class predictions is very low (<30%).     


## Conclusions

This was a harder problem than we thought. Reducing the number of labels probably would have helped, as would evaluating the images in context with other images. One of the papers we saw early on (Improving Transfer Learning for Movie Trailer Genre Classification using a Dual Image and Video Transformer) used a dual approach with video and frames, but we decided to compare frames and audio instead. Now it’s clear that the added context provided by the video helped their model a lot.

For audio, it’s worth considering how appropriate our models were for this task.  We felt comfortable with our assumption that trailers consisted of a series of audio events, but trailers obviously include more than that - dialogue and music are both very important tonally, and we are not using that information in the best way that we can.  A stronger approach would be something like a multimodal approach, consisting of audio event models as well as speech recognition and music classification ones.

Genre itself is subjective - which is maybe a cop-out, but looking at the genres in our dataset it is arguable that some of the genres exist primarily as subgenres, or else as genres that are primarily seen in conjunction with other, more popular, genres.  Limiting this analysis to popular genres (maybe something like Action, Comedy, Drama, Horror, Sci-Fi) would likely yield better performance.

Finally, multilabel problems are hard.  Good performance in a multilabel task requires more nuanced approaches.  We knew attempting this analysis was a bit ambitious, but felt that a multilabel problem would be a fun and interesting challenge for us.

Although performance was not great in either the image or audio portions of this problem, we have learned a lot about working with data of both types, and about working with movie audio in general. We also built a solid pipeline for extracting information from movie trailers in case anyone else wants to try and tackle this problem.
