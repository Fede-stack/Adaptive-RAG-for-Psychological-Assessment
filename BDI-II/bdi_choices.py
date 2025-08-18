bdi_items = [
    # 1. Sadness
    [
        "I do not feel sad.",
        "I feel sad much of the time.",
        "I am sad all the time.",
        "I am so sad or unhappy that I can't stand it."
    ],
    # 2. Pessimism
    [
        "I am not discouraged about my future.",
        "I feel more discouraged about my future than I used to be.",
        "I do not expect things to work out for me.",
        "I feel my future is hopeless and will only get worse."
    ],
    # 3. Past Failure
    [
        "I do not feel like a failure.",
        "I have failed more than I should have.",
        "As I look back, I see a lot of failures.",
        "I feel I am a total failure as a person."
    ],
    # 4. Loss of Pleasure
    [
        "I get as much pleasure as I ever did from the things I enjoy.",
        "I don't enjoy things as much as I used to.",
        "I get very little pleasure from the things I used to enjoy.",
        "I can't get any pleasure from the things I used to enjoy."
    ],
    # 5. Guilty Feelings
    [
        "I don't feel particularly guilty.",
        "I feel guilty over many things I have done or should have done.",
        "I feel quite guilty most of the time.",
        "I feel guilty all of the time."
    ],
    # 6. Punishment Feelings
    [
        "I don't feel I am being punished.",
        "I feel I may be punished.",
        "I expect to be punished.",
        "I feel I am being punished."
    ],
    # 7. Self-Dislike
    [
        "I feel the same about myself as ever.",
        "I have lost confidence in myself.",
        "I am disappointed in myself.",
        "I dislike myself."
    ],
    # 8. Self-Criticalness
    [
        "I don't criticize or blame myself more than usual.",
        "I am more critical of myself than I used to be.",
        "I criticize myself for all of my faults.",
        "I blame myself for everything bad that happens."
    ],
    # 9. Suicidal Thoughts or Wishes
    [
        "I don't have any thoughts of killing myself.",
        "I have thoughts of killing myself, but I would not carry them out.",
        "I would like to kill myself.",
        "I would kill myself if I had the chance."
    ],
    # 10. Crying
    [
        "I don't cry anymore than I used to.",
        "I cry more than I used to.",
        "I cry over every little thing.",
        "I feel like crying, but I can't."
    ],
    # 11. Agitation
    [
        "I am no more restless or wound up than usual.",
        "I feel more restless or wound up than usual.",
        "I am so restless or agitated that it's hard to stay still.",
        "I am so restless or agitated that I have to keep moving or doing something."
    ],
    # 12. Loss of Interest
    [
        "I have not lost interest in other people or activities.",
        "I am less interested in other people or things than before.",
        "I have lost most of my interest in other people or things.",
        "It's hard to get interested in anything."
    ],
    # 13. Indecisiveness
    [
        "I make decisions about as well as ever.",
        "I find it more difficult to make decisions than usual.",
        "I have much greater difficulty in making decisions than I used to.",
        "I have trouble making any decisions."
    ],
    # 14. Worthlessness
    [
        "I do not feel I am worthless.",
        "I don't consider myself as worthwhile and useful as I used to.",
        "I feel more worthless as compared to other people.",
        "I feel utterly worthless."
    ],
    # 15. Loss of Energy
    [
        "I have as much energy as ever.",
        "I have less energy than I used to have.",
        "I don't have enough energy to do very much.",
        "I don't have enough energy to do anything."
    ],
    # 16. Changes in Sleeping Pattern
    [
        "I have not experienced any change in my sleeping pattern.",
        "I sleep somewhat more than usual OR I sleep somewhat less than usual.",
        "I sleep a lot more than usual OR I sleep a lot less than usual.",
        "I sleep most of the day OR I wake up 1-2 hours early and can't get back to sleep."
    ],
    # 17. Irritability
    [
        "I am no more irritable than usual.",
        "I am more irritable than usual.",
        "I am much more irritable than usual.",
        "I am irritable all the time."
    ],
    # 18. Changes in Appetite
    [
        "I have not experienced any change in my appetite.",
        "My appetite is somewhat less than usual OR My appetite is somewhat greater than usual.",
        "My appetite is much less than before OR My appetite is much greater than usual.",
        "I have no appetite at all OR I crave food all the time."
    ],
    # 19. Concentration Difficulty
    [
        "I can concentrate as well as ever.",
        "I can't concentrate as well as usual.",
        "It's hard to keep my mind on anything for very long.",
        "I find I can't concentrate on anything."
    ],
    # 20. Tiredness or Fatigue
    [
        "I am no more tired or fatigued than usual.",
        "I get more tired or fatigued more easily than usual.",
        "I am too tired or fatigued to do a lot of the things I used to do.",
        "I am too tired or fatigued to do most of the things I used to do."
    ],
    # 21. Loss of Interest in Sex
    [
        "I have not noticed any recent change in my interest in sex.",
        "I am less interested in sex than I used to be.",
        "I am much less interested in sex now.",
        "I have lost interest in sex completely."
    ]
]


sentences_bdi = list(itertools.chain.from_iterable(bdi_items))
