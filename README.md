# Chord Identifier
Helping musicians play popular music.

Tristan Miller

Try the app at [autoguitartabs.xyz](http://autoguitartabs.xyz)!

To learn more, see [my slides](http://bit.ly/2ocNmn6)!

Many musicians at all levels of expertise are interested in playing their favorite songs, but one of the most difficult steps is identifying the correct chords.  My app will take in a link to any YouTube video, and analyze the audio to identify chords.

## Caveats
The app may take 15 seconds or more to analyze a new song.  Chord identification is a very difficult task, and the app is not very reliable!  The app provides buttons that will play back the chords so that you can inspect them for quality.  Although the repository and app url refer to guitar tabs, I ultimately decided not to include guitar fingerings, only the names of the chords.

## About this project
This was a 3-week project for *[Insight Data Science](https://www.insightdatascience.com)* in September 2019.  But other people have spent much longer trying to tackle the same problem.  Interested users may check out [MIREX](https://www.music-ir.org/mirex/wiki/MIREX_HOME), an annual competition that challenges academics to identify chords and extract other information from music, and [Chordify](https://chordify.net), a commercial app with the same goal as my own.

## About this repository
This repository contains my code to characterize data, train a machine learning model, and run a server.  However, I have excluded all data.  Chord data was downloaded from [MIREX](https://www.music-ir.org/mirex/wiki/2019:Audio_Chord_Estimation), and mp3 data was collected from YouTube.  If you are a developer interested in more information about what I did, please contact me.
