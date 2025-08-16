## Proof of concept anticheat implementation using machine learning to detect aimbot.
https://www.youtube.com/watch?v=s50n1qt9JnI

## Project Directory explanation:
- data: csv files for training and testing
- gameFilesC++: the game files used for the unreal engine. This includes the anti-memory manipulation software, as well as sending telemetry data. 
- models: the testing and training python files. These implement pytorch to train and test the LSTM model
- static: the HTML page served to display the models predictions
- guide.py: unecessary file of a tutorial followed to learn about models
- telemetryAPI.py: The initial file for data collection that served as a FastAPI server. This served as a connection for the database and the unreal engine.
- telemetryLiveAPI: The live version of the api that serves a static html, and does feature augmentation before in real time feeding data into the model
- 
## Report:
Machine Learning Anti-Cheat Software
William Craske 300180386
Department of Computing, University of the Fraser Valley
COMP 440
Dr. Omer Waqar
August 16th, 2025
Footage of software: https://youtu.be/s50n1qt9JnI
Github with full code: https://github.com/wcraske/MachineLearningAntiCheat


Chapter 1 - Fundamentals of Video Games, Anti-Cheats, and Machine Learning:
As long as video games have been around, malicious actors have found ways to exploit
the games. These actors can be defined as cheaters, who, in some capacity, are gaining an unfair
advantage. The methods these cheaters use depend on the style of game, such as an online
competitive game or a single-player offline game. Cheating can include modifying game
memory, injecting code, or using third-party tools to manipulate the game to provide an
advantage. To guard against these malicious actors, video game studios have started
implementing anti-cheat software, a software designed to deter and prevent hackers from
exploiting known cheating methods. Anticheats are more commonly found in online, multiplayer
games to ensure competitive integrity. These Anticheats use server-side checks to verify in real
time local and server-side states to ensure exploits are prevented. If a cheat is detected, the
malicious actor can be removed from the game as soon as possible, and in some cases,
mid-game. If the game is offline, anticheats are harder to enforce, but may be easier using
machine learning to detect heuristics and behavioural patterns of users. These patterns are
determined by the telemetry data of the user. Telemetry data refers to data collected from the raw
inputs a player generates while interacting with the game, as well as the in-game outcomes of
those actions. This can include mouse movements, key presses, player position changes, and the
coordinates of shot locations. By capturing this information, the player’s behaviour patterns
become apparent. A common form of cheating in first-person shooter games is known as an
“aimbot”. This is when a script or hack controls the user’s aim and points the player directly
facing the enemy, providing a competitive edge. My project aims to implement an anti-cheat
using machine learning to detect cheaters or hackers, as well as common anti-cheat
implementations to further deter and prevent malicious actors in a video game. This video game
will be developed using Unreal Engine 4, following the official first-person shooter game
tutorial.


Chapter 2: Literature Review:
Current literature on competitive integrity, cheating, and hacking in video games
indicates that there are many forms of malicious actors. These methods involve aimbots,
triggerbots, memory modification, wallhacks, and more. The way modern anti-cheat software is
headed is towards kernel-level software, along with server-side checks. These mechanisms help
prevent cheaters, but come with privacy concerns. Kernel-level software functions by accessing
the deepest layer of the operating system. This allows the software to have complete control and
access to every process running, to prevent malicious activity. Server-side check is another
strong approach, validating local data throughout the server, preventing users from tampering
locally to impact online gameplay. This reduces client tampering risks but requires infrastructure
to process and verify data in real time. Many modern systems combine many approaches for a
robust anti-cheat system.
In terms of machine learning, frameworks like PyTorch and TensorFlow are extremely popular in
the modern space. Both libraries utilize tensors, support GPU acceleration, and offer automatic
differentiation, enabling the efficient training and deployment of complex neural networks. The
library provides various data preprocessing utilities and integration options with other tools.
These are ideal for rapid testing and training, with iterative improvements being made quickly
and easily.
Game development engines like Unity and Unreal Engine equally provide accessible game
creation options, each with its advantages. Unreal offers advanced higher quality visuals, and can
handle larger scale games, utilizing C++ for the programming language. This offers a large
amount of specific customizability. Unity emphasizes ease of use for small developers, with a
large asset store, and uses C# for the coding base. Both engines are strong in their own right.


Chapter 3: Game Development:
To implement an anti-cheat, a developed game is a prerequisite. Following the official
Unreal First Person Shooter Tutorial, provided in their documentation, the tutorial walks the user
through the basics of game development and using the Unreal Engine. Unreal utilizes the
language C++, and by default connects to the IDE Visual Studio 2022. After the initial setup of
creating a project, character creation was necessary to provide the user with a medium to
experience the game. Input mappings for directional and camera movement were then
implemented for basic character movements, jumping, and turning the camera. These are simple
controls to allow the user to control the character with “WASD”, the space bar, and the mouse,
respectively. Then a mesh, a visible set of arms, was added to the character so the user is able to
see themselves. This mesh is blocked from visibility, so whenever viewing not from the
character’s perspective, it is invisible.
Then shooting mechanics were implemented. The tutorial implements a projectile-based
shooting, followed by collision. The temporary objects in the map would interact and be moved
by the collision. The projectiles would then despawn after a preset time to save memory. To aim
the shots, a heads-up display class was implemented with a crosshair in the center of the screen.
Lastly, character animation was implemented for the character. This included animating both
movement, jumping and idle states. For stylization, an Egyptian-style map and a weapon asset
were imported, as these flesh out the game.
This game development tutorial provided the foundational framework to implement an anti-cheat
system, with some changes and enhancements. The first change was to add an enemy to shoot.
The enemy class finds the controlling player and navigates toward them at a set speed. The same
skeletal mesh is provided to the enemy, so they are visible to the player. The enemy is given a
health attribute as well, so the player is able to defeat them after a few shots. These enemies are
spawned at periodic intervals by a separate enemy spawner class.
Then it was necessary to augment the HUD class, adding an ammo counter. This ammo counter
does not impact the gameplay, as there are no reload animations or pauses; it simply acts as an
on-screen text that displays a value that could be changed through hacking or adjusting memory
values. The counter starts at 30, and each shot decrements by one.
To simplify data collection, the current projectile-based system was replaced with hitscan, where,
instead of shooting a projectile that has “airtime”, hitscan sends an instantaneous “laser” from
the center of the player's screen outwards. This will matter later when machine learning is
implemented to simplify potential complexities.
Screenshot of the player shooting hitscan lasers into the sky.
The next changes are all made to the player class and are extensive. When adding additional
functions to this class, the main goal was to be able to provide telemetry data specific to aiming
and shooting at the enemy for the machine learning aspect. Telemetry data as touched on before,
is the data that is procured when the user interacts with the game. This can range from mouse
and keyboard inputs to the angle at which the user is looking. After reworking the shooting
system to be hitscan, the fire function, which controls this, deals damage to the enemy upon
raycast contact. Then, to send the telemetry data of the shot, the SendTelemetry function was
implemented, which delivers a payload to my local server. This function is called whenever the
player shoots and hits an enemy. The payload is sent via an HTTP POST request to an api
endpoint, and is formatted as JSON. The data that is saved to my local server is the player ID, the
time of the shot, the pitch and yaw of the camera, and the x, y, and z coordinates relative to the
enemy. This provides the means to harvest the data to form a “legitimate” dataset, with gameplay
from a non-malicious actor.
To procure the malicious actor gameplay, an “aimbot” class was developed. Aimbot, as described
before, is a cheat whereby the user can instantaneously, consistently aim and shoot at the target
with extremely high accuracy. In other words, the cheat finds the enemy, and retargets the player
camera to always face and follow the enemy. The aimbot function finds the closest enemy to the
player, gets the enemy's location, and rotates the player’s camera to always be locked onto the
enemy’s center point. Initially, the aimbot shot every frame as soon as possible, then shifted
towards shooting at realistic random intervals to simulate a human player. The “aimbot” dataset
was then filled with these two kinds of cheating telemetry data. The data is enqueued into a
FString to temporarily store the data before being sent at periodic time intervals, to avoid
overloading the server with data. This was necessary in the event that data was being sent every
frame, or extremely quickly.


Chapter 4: Initial Telemetry Collection and Processing:
Prior to capturing the data from the player class post request, a method to receive and
store the data was necessary. To solve this, a simple FastAPI application was developed. The
application receives the shot data from the Unreal Engine client in real time, and receives the
data. The application functions by instantiating a local SQLite database upon running, opening
up port 8000, and effectively listening for incoming POST requests. The database stores the pitch
and yaw of the player camera, the x, y, and z of where the shot landed on the enemy. Then the
timestamp is saved and added to the database. This provides a real-time gathering of shot data.
This data, along with the timestamp, can effectively recreate how the player is shooting the
enemy and can display behavioural patterns.
To make accessing the data easier, a few endpoints were created for the end user to view or
download the data. These are viewed as the raw data in JSON format, or when downloading, the
JSON is parsed and downloaded as a CSV file. This is done to allow for easy feeding into the
machine learning model, which comes later. Lastly, a static HTML page is served to display the
machine learning results, which will be discussed later.
A screenshot of the FastAPI Endpoint docs, with telemetry data.
The server’s role is to functionally support both data collection for legitimate and aimbot-assisted
gameplay. This data is concatenated and marked as either aimbot or legitimate to aid in
supervised machine learning, a critical step in building a live, real-time anti-cheat system.
Overall, the application provides a simple backend for telemetry data capture that will be
processed by the implemented machine learning model.


Chapter 5: Machine Learning:
To further enhance the cheat detection software and utilize the collected telemetry data to
determine cheating patterns, a machine learning model is necessary to implement. To properly
analyze time-related patterns, the model uses a Long Short-Term Memory (LSTM) architecture,
designed for sequence modelling. When analyzing individual rows of this data one at a time,
every shot on the enemy appears to be legitimate. There is no discrepancy between an honest
player and a malicious actor. LSTM is the perfect match when analyzing these patterns, as it can
learn behaviours over sections of data, such as unnatural aim adjustments or consistent shot
timing. Prior to passing the data into the model for training, the data needs to be normalized, then
sequenced into groups of 10 based on the timestamp, then split into three different sets; training,
validation and testing. Cross-validation was implemented for temporal separation between
training and evaluation data. This was to reduce overfitting to specific time segments. To further
add features that aid in temporal analysis, determining variance and rate of change was necessary
to be analyzed. In each group, for each chosen feature, the function calculates the rolling
variance, rolling mean, and standard deviation. Then, the delta values for the aim offset angles
are calculated to determine acceleration in skew, which is an indicator of player legitimacy.
Adding further model integrity, the LSTM is bidirectional, processing data in both forward and
backward directions, to determine player aim and behaviour. Further reduction of overfitting
included layer normalization and dropout layers. These stabilize training and reduce overfitting.
The output layer produced raw logits, which were converted to probabilities that represent how
probable the aimbot was being used. During data collection, the data was skewed to have more
aimbot values, so to correct this imbalance, a weighted loss function was implemented. This
ensures the model equally values both the legitimate and aimbot aspects of the dataset. The
model was trained using the adaptive moment optimizer, which uses momentum and RMSprop
to adjust learning rates during training. Performance was measured using the F1 score, which
values both precision and recall.
The model was finally evaluated on a portion of the test set that was not introduced during
training. It outputs confusion matrices, which indicate the performance for predicted values that
are false positives, true positives, false negatives, and true negatives. The results of the training
indicated a confidence of 95%, which indicated either that the model is overfitting the data or the
data is extremely obvious. The final model was then saved, and awaited testing on new data.
Before implementing the model into a real-time anticheat, the trained model was evaluated
against new telemetry data that was gathered. This was to ensure that the model was not
overfitting the initial data. The testing essentially replicated the same preprocessing methods
from the training portion. This included rolling statistics, variance, acceleration metrics for aim
offsets, and temporal deltas. These were then normalized similarly to avoid distribution shifts.
When the model came to the prediction stage, there was a threshold of probability necessary to
classify each group as legitimate or aimbot behaviour. These predictions came out with a high
confidence, at approximately 85% confidence over multiple tests. This indicates that the model is
ready to be integrated into the real-time anti-cheat software.
Chapter 6: Real-Time Machine Learning:
Since the model is trained and performs accurately on fresh data, the next step was to
integrate the model into our FastAPI application. This would allow the game to communicate in
a real-time environment. This was built off the initial FastAPI application that allowed the game
to communicate and send telemetry data to the local server, with the model loaded and some
added feature augmentation.
Along with the database storing telemetry data, the backend also serves a static HTML page that
displays the live predictions of the model. This page periodically updates every few minutes and
displays a table that holds the confidence level of the model’s belief that the user is aimbotting.
These updates are scheduled in the background and query the last five minutes of telemetry data.
After retrieving the data, the same features are calculated as before: the rolling variance, rolling
mean, and delta offsets. The most recent prediction per player is stored in the predictions table,
along with the probability score and is accessible via an endpoint.
Screenshot of the static HTML served to display predictions of the model.
This anti-cheat system functions in intervals of approximately 3 minutes, which is the average
time the model spends predicting based on the available data. It runs continuously, with no
manual inputs required. If this were deployed as a functional anti-cheat, features could be added
to flag the player or kick them from the game.
Chapter 7: Anti-Cheat Implementation:
Then an anti-cheat class was created. This class manages a few common cheat deterrence
methods, such as memory manipulation and using common cheat programs. To detect common
cheat programs, the anti-cheat employs a function called EnumProcesses. This function is a
Windows function and is implemented using a specific library. EnumProcessModules is then
called to get each process name and compare it to a list of known cheating software, such as
“Cheat Engine”. If a match is found in the process names, the anticheat produces an in-game
warning, and logs this onto the screen.
Along with scanning for processes, the anti-cheat software also uses XOR encryption. As
mentioned previously, the variables “ammo count” are stored in memory using XOR encryption,
then with an added checksum to detect any memory tampering. As a further implementation for
extra security, a shadow variable is introduced, which should have the same value when
compared to the original ammo count. This is checked periodically to ensure that there are no
malicious actors tampering with memory.
Screenshot of game with manipulated ammo count with log “Memory manipulated, ammo count
does not match”
When designing the anti-cheat software, deterrence and prevention are the main goals. The
methods implemented will prevent common cheating methods such as external memory
manipulation and the execution of known cheating software such as “Cheat Engine”. These
methods, combined with a live machine learning model, make an effective combination of
preventing tampering and catching those who have found exploits.
The primary feature of the system is detecting running processes associated with known cheating
tools. The function utilizes Windows Process Status API (PSAPI) via the EnumProcesses
function, which retrieves a list of process identifiers for all currently running processes on the
system. For each process ID returned, the class uses EnumProcessModules and
GetModuleBaseName to extract the executable name of the process. These process names are
then compared against a pre-made list of known cheat programs. If a match is found, a log is
output to the console as proof of concept.
Then, to further prevent tampering, XOR obfuscation and encryption are implemented. As a
visible, exploitable feature, the ammo count will be the variable that is encrypted, since it is
logged on the screen; any change in the memory will be displayed easily. Each ammo value is
stored in memory, encrypted with an XOR key, then stored with a checksum. The checksum is
used to further verify integrity. If the current memory checksum fails to match the stored
checksum, the system flags the discrepancy as a potential malicious activity and then logs it in
the console.
Lastly, the anti-cheat system maintains a shadow copy of the ammo count. The shadow count
simply acts as the real ammo counter would, decrementing after each shot and is periodically
compared against the real ammo variable. Any difference in these values is a possible memory
exploit.

Chapter 7: Pitfalls:
There were a few challenges during the development of the anti-cheat system. The initial
challenge was determining the sort of data that was necessary to collect for a binary decision
machine learning model. Previously, I had assumed that only legitimate telemetry was necessary
for the model, but this was not the case. The model required both legitimate and cheated data to
learn the patterns of both behavioural trends. To accomplish this, an aimbot feature was added to
the game, where the telemetry for aimbot could be collected. Initially, the aimbot fired a shot
every frame, resulting in a server overload. To fix this, the enqueue system was implemented,
periodically sending short data over instead of instantaneously. After data collection was
successful, I initially implemented a simple binary classifier with both a wide and a deep model.
When testing, the model’s accuracy was 47%, worse than flipping a coin to determine if the
player is aimbotting. After further research, the solution to this was to implement a Long
Short-Term Memory model. This model focuses on temporal sequences, as opposed to specific
features on a per-row basis.

Chapter 8: Results:
The combined anti-cheat system from both the deterrence system, anti-memory
tampering, and the real-time machine learning model makes for a strong guard against malicious
act
ors. Breaking down the individual components, the process scanning is an effective tool given
that the predetermined list is comprehensive. This can be bypassed if the process is given a new
name, but it will cover generic cases. Memory protection, such as XOR encryption, checksums,
and shadow variable verification, is a consistent and standard countermeasure for an anti-cheat
system. When testing on my game, altering the memory values for the ammo counter are
consistently flagged, due to the redundancy of measures in place.
The machine learning model performed strongly in identifying users with aimbot in real time.
When playing with an aimbot, the machine learning frequently predicted correctly, with a
confidence of around 85%. When playing legitimately, the aimbot could easily detect it. This
suggests a potential change in the threshold for the model to be more sensitive when predicting
aimbot. The model struggled to identify aimbot during transition phases, starting by playing
legitimately, then switching on aimbot temporarily. This is the result of the prerequisite of
gathering the past five minutes of data. These false negatives suggest that the data be evaluated
faster with smaller groups of data.
Overall, the integrated anti-cheat system demonstrates relatively high effectiveness in both
prevention and detection. Combining memory manipulation and process deterrence alongside
real-time anti-cheat software provides a strong defence against common cheating methods.


References:
Ketkar, N., & Moolayil, J. (2021). Introduction to PyTorch. In Deep Learning with Python.
Apress. https://doi.org/10.1007/978-1-4842-5364-9_2
Lehtonen, S. (2020, March 7). Comparative study of anti-cheat methods in video games
[Bachelor’s thesis, University of Helsinki]. Helda.
https://helda.helsinki.fi/server/api/core/bitstreams/89d7c14b-58e0-441f-a0de-862254f95551/cont
ent
Sanders, A. (2016). An introduction to Unreal Engine 4 (1st ed.). A K Peters/CRC Press.
https://doi.org/10.1201/9781315382555
Drachen, A. (2015). Behavioral telemetry in games user research. In R. Bernhaupt (Ed.), Game
user experience evaluation (pp. [chapter pagination unknown]). Springer.
https://doi.org/10.1007/978-3-319-15985-0_7
Epic Games. (n.d.). Implementing your character (Unreal Engine 4.27 documentation). Epic
Games.
https://dev.epicgames.com/documentation/en-us/unreal-engine/2---implementing-your-character?
application_version=4.27
Shawnthebro. (2021, May 24). UE4 Blueprint: Third person game creation (part 5) [Video].
YouTube.
https://www.youtube.com/watch?v=wI3Hj3_cUFQ&list=PLfAjixzz6o82cLilSI3Vbv0hu868hAp
7P&index=5
PyTorch. (n.d.). Quickstart tutorial (Beginner basics). PyTorch.
https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
