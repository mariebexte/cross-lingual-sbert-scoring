import pandas as pd
import os, sys
import copy
import time
import gc
from easynmt import EasyNMT
import nltk
from torch.cuda import OutOfMemoryError
import torch


# Translate ASAP into all ePIRLS languages

nltk.download('punkt')

data_path = '/data/ASAP/split'
other_languages = ['ar', 'da', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

# Language codes for translation model
language_codes = {'ar': 'ar', 'da': 'da', 'en': 'en', 'he': 'he', 'it': 'it', 'ka': 'ka', 'nb': 'no', 'pt': 'pt', 'sl': 'sl', 'sv': 'sv', 'zh': 'zh'}

translation_model = EasyNMT('m2m_100_1.2B', max_loaded_models=1)

for prompt in ['1','2','3','4','5','6','7','8','9','10']:

    print(prompt)

    for split in ['train.csv', 'val.csv', 'test.csv']:

        # All original data is English
        df = pd.read_csv(os.path.join(data_path, prompt, 'en', split))

        for other_language in other_languages:

            if not os.path.exists(os.path.join(data_path, prompt, other_language, split)):

                if not os.path.exists(os.path.join(data_path, prompt, other_language)):

                    os.mkdir(os.path.join(data_path, prompt, other_language))

                df_copy = copy.deepcopy(df)
                df_copy['AnswerText'] = df_copy['AnswerText'].apply(str)
                
                ### to debug
                translated_texts = []
                original_texts = list(df_copy['AnswerText'])

                for text in original_texts:

                    if text == 'Paul is a student that has trouble reading and he takes part in a remedial reading program that allows him to get help on reading a couple of times a week.  he would leave english early to go to this program and every time Mr. Leonard saw him he would ask were he was going and check his hall pass so that he could make sure that he was going to the right place.     When Mr. Leonard asked Paul to stay after school one day and meet him in the gym Paul said ok becuase he would take any route that didnt lead to doing homework.  Once Paul got to the Gym he was asked by Mr. Leonard whether he enjoyed basketball and he nodded his head not so he took him outside to the track and had him jump over some hurdles.  When he was finished he said that was aweful but Mr. Leonard said that it was good and he would get better as he practiced so he told Paul to show up tommorow with running shows and a pair of shorts.     I began to wonder why Mr. Leonard was helping him like this and once he met the high school track coach and he told him to look Mr. Leonard up he found out that he was a very accomplished track runner back in college and he fluncked out of college becuase the scouts did not care about grades and so he had to do harder work and couldnt do it even with the tutors help.       Paul began to realize that Mr. Leonard and himself are alot alike and he wanted to help a child reach his potential athletically, since Paul and already sawt help in his learning troubles.':
                        
                        translated_texts.append('პოლი არის სტუდენტი, რომელსაც კითხვა უჭირს და ის მონაწილეობს კითხვის გამოსასწორებელ პროგრამაში, რომელიც საშუალებას აძლევს მას კვირაში რამდენჯერმე კითხვის საკითხში დაეხმაროს. ის ადრე ტოვებდა ინგლისურს ამ პროგრამაზე წასასვლელად და ყოველ ჯერზე, როცა მისტერ ლეონარდს ხედავდა, ეკითხებოდა, მიდიოდა თუ არა და ამოწმებდა მის დარბაზს, რათა დარწმუნდა, რომ სწორ ადგილას მიდიოდა. როდესაც მისტერ ლეონარდმა პოლს სთხოვა, რომ ერთ დღეს სწავლის შემდეგ დარჩენილიყო და სპორტდარბაზში შეხვედროდა, პოლმა თქვა კარგი, რადგან ის წავიდოდა ნებისმიერ მარშრუტზე, რომელიც არ გამოიწვევს საშინაო დავალების შესრულებას. როგორც კი პოლ სპორტდარბაზში მივიდა, მას ბ-ნმა ლეონარდმა ჰკითხა, უყვარდა თუ არა კალათბურთი და თავი დაუქნია, ისე რომ გარეთ გაიყვანა ტრასაზე და გადახტა რამდენიმე ბარიერზე. როცა დაასრულა, თქვა, რომ საშინელება იყო, მაგრამ მისტერ ლეონარდმა თქვა, რომ ეს კარგი იყო და ვარჯიშის დროს უკეთესები გახდებოდა, ამიტომ უთხრა პოლს, რომ ხვალ გამოჩენილიყო სირბილით და შორტებით. დავიწყე ფიქრი, რატომ ეხმარებოდა მას ასე ბატონი ლეონარდი და როგორც კი შეხვდა საშუალო სკოლის ტრეკის მწვრთნელს და უთხრა, რომ მისტერ ლეონარდს აეხედა, აღმოაჩინა, რომ ის იყო ძალიან წარმატებული ტრასის მორბენალი კოლეჯში და გაიქცა. კოლეჯში, რადგან სკაუტებს არ აინტერესებდათ ქულები და ამიტომ მას უფრო რთული სამუშაოს შესრულება მოუწია და მასწავლებლების დახმარებითაც კი ვერ ახერხებდა ამას. პოლმა დაიწყო იმის გაცნობიერება, რომ მისტერ ლეონარდი და თვითონაც ძალიან ჰგვანან ერთმანეთს და მას სურდა დაეხმარა ბავშვს თავისი პოტენციალის სპორტულად მიღწევაში, რადგან პოლ და უკვე ნახა დახმარება სწავლის პრობლემებში.')
                        print('APPENDED DEEPL')

                    elif text == "When reading the article, the reader first sees Mr. Leonard as a kind of mean guy, that just likes to heard people along in the hallways at schoool, because he is a hall monitor. Later on in the story when Mr. Leonard takes Paul under his wing to teach him how to be a good runner the reader learns that Mr.Leonard used to be a very succesful track runner, and used to be very well known. The reader also learns that he flunked out of college, and that's why he is no longer a runner.      Paul is a kid that can't read, and Mr. Leonard seemed to take an interest in him and was going to help him learn how to be a good runner, because he thinks it will help him out. Paul listens to Mr. Leonard without any hesitation and quickly learns to run very well. In the process of joining the high school track team Paul learns that Mr. Leonard used to be a very well known track star. This confuses Paul on why Mr. Leonard didn't tell him this.     When Paul asks why he wasn't told he learns that Mr. Leonard didn't tell him because he didn't want to brag about how good he used to be, and that he was also embarrased because he had flunked out of school because he couldn't read.          The fact the Mr. Leonard took Paul under his wing to help him accomplish things in life better, and that Mr. Leonard wanted to help Paul because he never got any help tells the real type of mean Mr. Leonard is and Paul sees that and is very greatful.":
                        
                        translated_texts.append('სტატიის კითხვისას მკითხველი ჯერ მისტერ ლეონარდს ხედავს, როგორც ერთგვარ ბოროტ ბიჭს, რომელსაც უბრალოდ მოსწონს ხალხის მოსმენა სკოლის დერეფნებში, რადგან ის არის დარბაზის მონიტორი. მოგვიანებით მოთხრობაში, როდესაც მისტერ ლეონარდ პოლს ასწავლის, თუ როგორ უნდა იყოს კარგი მორბენალი, მკითხველი გაიგებს, რომ ბატონი ლეონარდი ადრე ძალიან წარმატებული ტრასაზე მორბენალი იყო და ძალიან ცნობილი იყო. მკითხველი იმასაც იგებს, რომ კოლეჯიდან გავარდა და ამიტომ მორბენალი აღარ არის. პოლი არის ბავშვი, რომელსაც კითხვა არ შეუძლია და მისტერ ლეონარდს, როგორც ჩანს, დაინტერესდა მისით და აპირებდა დაეხმარა მას ესწავლა კარგი მორბენალი, რადგან ფიქრობს, რომ ეს მას დაეხმარება. პოლი ყოველგვარი ყოყმანის გარეშე უსმენს მისტერ ლეონარდს და სწრაფად სწავლობს ძალიან კარგად სირბილს. საშუალო სკოლის ტრეკის გუნდში გაწევრიანების პროცესში პოლ იგებს, რომ მისტერ ლეონარდი ადრე ძალიან ცნობილი ტრეკის ვარსკვლავი იყო. ამან პოლს აბნევს, რატომ არ უთხრა მისტერ ლეონარდმა ეს. როდესაც პოლი ეკითხება, რატომ არ უთხრეს, გაიგებს, რომ მისტერ ლეონარდს არ უთქვამს, რადგან არ სურდა ეკვეხნა, რამდენად კარგი იყო ადრე და ისიც უხერხული იყო, რადგან სკოლიდან გაიქცა. მან ვერ წაიკითხა. ის ფაქტი, რომ მისტერ ლეონარდმა აიღო პოლი თავის ფრთების ქვეშ, რათა დაეხმარა მას ცხოვრებაში უკეთესად მიეღო საქმეები, და რომ მისტერ ლეონარდს სურდა დახმარებოდა პოლს, რადგან მას არასოდეს მიუღია დახმარება, მეტყველებს იმაზე, თუ როგორია ბატონი ლეონარდი და პოლი ხედავს ამას და ძალიან დიდებულია.')
                        print('APPENDED DEEPL')

                    elif text == "The background information that the reader recieves about Mr. Leonard is that he was a track star through high school and his freshman year in college, but that he didn't know how to read. His grades were terrible and he flunked out of college despite being an incredible runner, it didn't help him academically. One of the best runners in history who broke records had more in common with Paul than he could have ever imagined. Paragraph 45 states, 'The emotions in Mr. Leonard's words were all too familiar to me. I knew them well-feeling of embarrassment when I was called upon to read aloud or when I didn't know an answer everyone else knew. This man had given his time to help me excel at something. Suddenly I realized what I could for him.' Paul realizes that Mr. Leonard also cannot read but took his time out of his day to teach Paul how to excel at something that he did know how to do, and knew how to do it very well. Paul returns this favor to the best of his ability by trying to help Mr. Leonard learn how to read. Paragraph 46 says, 'C'mon, Mr. Leonard, I said, walking back toward the school. It's time to start your training.'":
                        
                        translated_texts.append("ფონური ინფორმაცია, რომელსაც მკითხველი იღებს მისტერ ლეონარდის შესახებ, არის ის, რომ ის იყო ტრეკის ვარსკვლავი საშუალო სკოლისა და კოლეჯის პირველ კურსზე, მაგრამ არ იცოდა კითხვა. მისი შეფასებები საშინელი იყო და მან მიატოვა კოლეჯი, მიუხედავად იმისა, რომ წარმოუდგენელი მორბენალი იყო, ეს მას აკადემიურად არ დაეხმარა. ისტორიაში ერთ-ერთ საუკეთესო მორბენალს, რომელმაც რეკორდები მოხსნა, პოლთან იმაზე მეტი საერთო ჰქონდა, ვიდრე წარმოედგინა. 45-ე პუნქტში ნათქვამია: „მისტერ ლეონარდის სიტყვებში არსებული ემოციები ჩემთვის ძალიან ნაცნობი იყო. მე კარგად ვიცოდი, რომ ისინი უხერხულ გრძნობას განიცდიდნენ, როცა ხმამაღლა წასაკითხად მეძახდნენ, ან როცა არ ვიცოდი პასუხი, რომელიც ყველამ იცოდა. ამ კაცმა დათმო თავისი დრო, რათა დამეხმარა რაღაცის გამორჩევაში. უცებ მივხვდი, რა შემეძლო მისთვის“. პოლი ხვდება, რომ მისტერ ლეონარდს ასევე არ შეუძლია კითხვა, მაგრამ დაუთმო დრო, რათა ესწავლებინა პოლს, როგორ გამოეჩინა ის, რისი გაკეთებაც მან იცოდა და ძალიან კარგად იცოდა. პოლ უბრუნდება ამ კეთილგანწყობას მისი შესაძლებლობების ფარგლებში და ცდილობს დაეხმაროს მისტერ ლეონარდს კითხვის სწავლაში. 46-ე პარაგრაფში ნათქვამია: „მოდით, მისტერ ლეონარდ, ვთქვი მე და სკოლისკენ მივდიოდი. დროა დაიწყოთ თქვენი ვარჯიში.'")
                        print('APPENDED DEEPL')

                    elif text == "The author first catches your attention using the introduction. In the introduction, the author says 'Grab your telescope! Look up in the sky! It's a comet! It's a meteor! It's a ... tool bag?'Then, the author explains what the item of the topic is and how it got there. The author says 'Over the past 52 years, a variety of spacecraft ... have been sent beyond Earth's atmosphere.' After that the author explains the consepuences of the item by saying 'with no one at the controls, dead satellites run the risk of colliding with each other.' the author follows up by saying 'Tiny fragments traveling at a speed of five miles per second can inflict serious damage on the most carefully designed spacecraft.' The author closes his article by explaining the way the situation is being taken care of. The author says 'Space agencies hope that the corporations and nations invelved can work together to come up eith a viable solution to space pollution.'":
                        
                        translated_texts.append("ავტორი პირველ რიგში იპყრობს თქვენს ყურადღებას შესავლის გამოყენებით. შესავალში ავტორი ამბობს: „აიღე შენი ტელესკოპი! აიხედე ცაში! ეს კომეტაა! ეს მეტეორია! ეს არის ... ხელსაწყოების ჩანთა?'შემდეგ, ავტორი განმარტავს, რა არის თემის პუნქტი და როგორ მოხვდა იქ. ავტორი ამბობს: 'ბოლო 52 წლის განმავლობაში, სხვადასხვა კოსმოსური ხომალდები ... გაიგზავნა დედამიწის ატმოსფეროს მიღმა'. ამის შემდეგ ავტორი ხსნის ნივთის შედეგებს იმით, რომ „არავის კონტროლზე, მკვდარი თანამგზავრები ემუქრებიან ერთმანეთს შეჯახების რისკს“. ავტორი შემდეგნაირად ამბობს: „პატარა ფრაგმენტებს, რომლებიც მოძრაობენ წამში ხუთი მილის სიჩქარით, შეუძლიათ სერიოზული ზიანი მიაყენონ ყველაზე ფრთხილად შემუშავებულ კოსმოსურ ხომალდს“. ავტორი ხურავს თავის სტატიას იმით, რომ განმარტავს სიტუაციის მოგვარების გზას. ავტორი ამბობს, რომ „კოსმოსური სააგენტოები იმედოვნებენ, რომ ჩართული კორპორაციები და ქვეყნები ერთად იმუშავებენ კოსმოსის დაბინძურების ეფექტური გადაწყვეტის მოსაძებნად“.")
                        print('APPENDED DEEPL')

                    elif text == "The author organizes the article into different sections each with a defined purpose. The first section is an attention grabber to get the reader interested. Line 1: 'Grab you telescope! look up in the sky! It's a comet! It's a meteor!' The second section is background to give the reader a little knowledge about the topic. Line 4: 'In 1957, the Soviet Union launched th efirst artificial satellite. The United States followed suit, and thus began the human race's great space invasion.' The third section is what's going on now with the topic. Line 6: 'With no one at the controls, dead satellites run th erisk o fcollding with each other. That's exactly what happened in February 2009.' The last section is why the reader should care about the topic. Line 8: 'So who cares about a lost tool bag or tiny bits of space trash?'":
                        
                        translated_texts.append("ავტორი აწყობს სტატიას სხვადასხვა განყოფილებებად, თითოეულს განსაზღვრული მიზნით. პირველი განყოფილება არის ყურადღების მიპყრობა მკითხველის დაინტერესების მიზნით. ხაზი 1: „აიღე ტელესკოპი! აიხედე ცაში! ეს კომეტაა! ეს მეტეორია!' მეორე ნაწილი არის ფონი, რათა მკითხველს ცოტაოდენი ცოდნა მისცეს თემის შესახებ. ხაზი 4: '1957 წელს საბჭოთა კავშირმა გაუშვა პირველი ხელოვნური თანამგზავრი. შეერთებულმა შტატებმა მიბაძა და ამით დაიწყო კაცობრიობის დიდი კოსმოსური შეჭრა.' მესამე განყოფილება არის ის, რაც ახლა ხდება თემასთან დაკავშირებით. სტრიქონი 6: „მკვდარი თანამგზავრების კონტროლის გარეშე, ერთმანეთთან შეჯახების საფრთხე ემუქრებათ. ზუსტად ასე მოხდა 2009 წლის თებერვალში.' ბოლო ნაწილი არის ის, თუ რატომ უნდა აინტერესებდეს მკითხველს თემა. სტრიქონი 8: 'მაშ, ვის აინტერესებს დაკარგული ხელსაწყოების ჩანთა ან კოსმოსური ნაგვის პატარა ნაჭრები?'")
                        print('APPENDED DEEPL')

                    elif text == "They begin the article with a catching introduction, drawing the reader to the article with a humourous quote. They add a twist to the quote: 'It's a . . . tool bag?'. This is to foreshadow the story of the accident in which a toolbag escaped the grip of an astronaut during a repair on the Internation Space Station. The author proceeds to explain what 'space junk' is, and how much of it is formed. The article explains the dangers of dead satellites that can crash into each other. It explains that with the increase of space exploration, there will be a collective increase of space junk. The author says that many people care about the space junk, offering opposition to readers who are indifferent to this issue. The author concludes that space agencies should work together to find a solution to this important problem.":
                        
                        translated_texts.append("ისინი იწყებენ სტატიას მიმზიდველი შესავლით, იუმორისტული ციტატით მიიპყრობენ მკითხველს სტატიისკენ. ისინი ციტატას ირონიას ამატებენ: „ეს არის . . . ხელსაწყოს ჩანთა?'. ეს არის ავარიის ამბავი, რომლის დროსაც საერთაშორისო კოსმოსურ სადგურზე რემონტის დროს ასტრონავტის ხელში ჩანთა გადაურჩა. ავტორი აგრძელებს იმის ახსნას, თუ რა არის „კოსმოსური ნაგავი“ და რა ნაწილი იქმნება. სტატიაში აღწერილია მკვდარი თანამგზავრების საშიშროება, რომლებიც შეიძლება ერთმანეთს დაეჯახოს. იგი განმარტავს, რომ კოსმოსური ძიების ზრდასთან ერთად, იქნება კოსმოსური ნაგვის კოლექტიური ზრდა. ავტორი ამბობს, რომ ბევრს აინტერესებს კოსმოსური ნაგავი, ოპოზიციას სთავაზობს ამ საკითხის მიმართ გულგრილი მკითხველს. ავტორი ასკვნის, რომ კოსმოსურმა სააგენტოებმა ერთად უნდა იმუშაონ ამ მნიშვნელოვანი პრობლემის მოსაგვარებლად.")
                        print('APPENDED DEEPL')

                    else:

                        translated_texts.append(translation_model.translate(text, source_lang=language_codes['en'], target_lang=language_codes[other_language]))
                    
                    df_copy['AnswerText'] = translated_texts + ([''] * (len(original_texts) - len(translated_texts)))
                    df_copy.to_csv(os.path.join(data_path, prompt, other_language, 'partial_' + split))

                df_copy.to_csv(os.path.join(data_path, prompt, other_language, split))
                print('Translated', prompt, other_language, split)
            
            else:
                
                print('Already Done', prompt, other_language, split)
