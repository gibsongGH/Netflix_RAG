import os
import chromadb
import uuid
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
COLLECTION_NAME = "netflix_culture"

CHUNK_SIZE = 1000
OVERLAP = 200    
MIN_CHUNK_SIZE = 400
N_RESULTS = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set — app will not function properly.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are a demo Netflix information assistant for a personal portfolio project. "
    "Answer only from the provided context. "
    "The context may include the Netflix Culture Memo, Netflix Work-Life Philosophy, "
    "and the Netflix 2025 Annual Report introduction. "
    "Do not invent policies, benefits, or company facts. "
    "If the answer is not clearly supported by the context, say you do not know. "
    "When helpful, answer clearly and concisely in a professional tone."
)   #"Do not add a sources section or citations in your answer. The application will display sources separately. "

# Global collection handle
collection = None

# ---------------------------------------------------------------------------
# Document https://jobs.netflix.com/netflix-culture.pdf
# ---------------------------------------------------------------------------
document_culture_memo = """
The Best Work of Our Lives
CULTURE MEMO
At Netflix, we aspire to entertain the world, thrilling audiences everywhere. To do that, we've
developed an unusual company culture focused on excellence, and creating an environment
where talented people can thrive — lifting ourselves, each other and our audiences higher and
higher. This document is about that culture, which is based on four core principles:
● The Dream Team: We aim only to have high performers at Netflix — people who are
great at what they do, and even better at working together.
● People over Process: You get better outcomes when employees have the information
and freedom to make decisions for themselves. We hire unusually responsible people
who thrive on this openness and freedom.
● Uncomfortably Exciting: To entertain the world, we need to be bold and ambitious.
That means embracing the thrill of what's next — even when it's uncomfortable.
● Great and Always Better: We often say Netflix sucks today compared to where we can
be tomorrow. We need the self-awareness to understand what should be better, and
the discipline and resilience to get there.
While we don't always live up to these principles, most people who join Netflix are pleasantly
surprised by how great their colleagues are and the way we empower people at every level.
As our business grows and evolves, our culture (and this document) will, too. What won't
change is our focus on excellence, and our determination to ensure that Netflix remains a
place where great people can do the best work of their lives. If this sounds exciting, you'll
probably love it here. But Netflix is not for everyone, so please read on.
THE DREAM TEAM
We believe that what makes a fantastic workplace isn't a great office or free meals and
massages — although we have some nice perks. It's the people. Imagine working alongside
stunning colleagues who are great at what they do, and even better at working together. It's
why we model ourselves on a professional sports team, not a family. Families are about
unconditional love. They can also be dysfunctional, as anyone who's watched Ozark or
Wednesday knows. Professional sports teams, on the other hand, focus on performance and
picking the right person for every position, even when that means swapping out someone
they love for a better player.
© 2024 Netflix, Inc. All rights reserved. | Last updated: June 2024 | English Version
While every member of our Dream Team has different skills, we look for common strengths
that make us better together. These are the values we value:
● Selflessness — you are humble when searching for the best ideas; you seek what's
best for Netflix, not yourself or your team; you take time to help others succeed.
● Judgment — you look beyond short term fixes in favor of long term solutions; you
make wise decisions despite ambiguity; you use data to inform your intuition.
● Candor — you willingly receive and give feedback; you are open about what's working
and what needs to improve; you admit mistakes openly and share learnings widely.
● Creativity — you welcome new ideas; you are passionate and persistent in pursuit of
more innovative solutions; you value artistic expression.
● Courage — you are vulnerable in search for the truth; you are willing to risk failure, or
challenge the status quo, in the pursuit of excellence.
●  Inclusion — you recognize your biases and work to counteract them; you try to ensure
everyone at Netflix can do their best work, whatever their culture, identity or
background.
● Curiosity — you learn rapidly and eagerly; you are more interested in other people's
ideas than your own; you're humble about what you don't yet know.
● Resilience — you quickly adapt to changing circumstances; you make tough decisions
without agonizing or long delay; you embrace a hard challenge.
It's easy to talk about values and harder to live by them. We all work hard to keep each other
accountable for upholding these standards, especially our leaders, because excellence and
honesty go hand in hand. It's why we invest in strong professional relationships that build trust
and help people assume good intent. This, in turn, enables us to practice extraordinary
candor — ensuring constructive feedback is part of our everyday work (like brushing your
teeth). It takes courage and vulnerability to ask someone how you could do better, or to seek
alternative opinions about the best way forward, and integrity only to say things about a
colleague that you're willing to share with them directly. This is especially true when you're
giving feedback to someone more senior or from a different background, or if you come from
a culture or company where deference is the norm. But extraordinary candor helps us improve
faster as individuals and a company.
Since a high performer in any role is many times more effective than the average employee,
our Dream Team is driven by performance — not seniority, tenure or unconditional loyalty. It's
also why we focus on maintaining a high performance culture. To recruit and retain stunning
colleagues, we pay personal top of market for the role and location — a judgment about what
that person could make in a similar role at another company, and what we would pay to keep
or replace them. We expect leaders to be strong developers of talent. And to ensure they have
the right player at every position, we ask them to apply what we call the "keeper test" —
asking "if X wanted to leave, would I fight to keep them?" Or "knowing everything I know
today, would I hire X again?" If the answer is no, we believe it's fairer to everyone to part ways
quickly.
In the abstract, the keeper test can sound scary. In reality, we encourage everyone to speak to
their managers about what's going well and what's not on a regular basis. This helps avoid
surprises. Managers also evaluate team members on their whole record, rather than focusing
on the mistakes or bets that didn't pay off. On the Dream Team, you need people who
challenge the status quo and try new things. So we stick with employees through short-term
bumps.
No matter how brilliant someone may be, there's no place in our Dream Team for people who
don't treat their colleagues with decency and respect. When you have talented people who
work well together — trusting each other's intentions and respecting their differences — it
makes everyone more successful.
PEOPLE OVER PROCESS
Many of us have worked at companies where decisions were made top down, there was little
transparency and it felt hard to make a difference — or even get basic things done. At Netflix,
we aim to inspire and empower more than just manage because people can have a greater
impact when they're free to make decisions about their own work.
As part of this, we strive to develop good decision-making muscles at every level of the
company, priding ourselves on how few, not how many, decisions senior leaders make. We
expect managers to practice context not control — giving their teams the context and clarity
needed to make good decisions instead of trying to control everything themselves. We also
help employees learn by sharing a lot of information internally, including through memos
where they can comment and ask questions. It takes an unusually responsible person to
thrive on this level of freedom — someone who's self-motivated, self-aware and
self-disciplined, who doesn't wait to be told what to do and picks up the trash like they would
at home. That said, context not control should not be confused with hands-off management.
Managers need to be involved in the work being done around them, and actively coach their
teams. They may also have to step in when someone is about to make a decision that is
unethical or could materially harm Netflix, during a crisis or if a new team member lacks the
full context.
We avoid decision-making by committee, which tends to slow companies down and
undermine accountability. For every significant decision, we identify an informed captain
who's responsible for making a judgment call on the right way ahead. Then different teams,
each led by their own informed captain, implement the decision. This highly aligned and
loosely coupled approach gives teams the freedom to move quickly and operate
independently, while ensuring responsibility for the outcome.
We've learned that the best ideas can come from anywhere, which is why we expect informed
captains to seek out different opinions and listen to people at every level. We call this farming
for dissent. Of course, not all opinions are created equal — and with 10,000-plus employees,
it's impractical for everyone to weigh in on most decisions (this memo being an exception). So
on an important creative decision, for example, the opinion of someone working in TV, film or
games will carry more weight than an engineer, and vice versa when it comes to our product
or technology. After a decision is made, we expect everyone, including the people who argued
for a different approach, to disagree and commit. This helps ensure the outcome is as
successful as possible. Afterwards, when the impact is clear, the informed captain should
reflect on their choices — what worked and what didn't — so everyone can learn how to do
better next time.
Launching a game, TV show or film, running a marketing campaign, managing compensation
and closing a quarter all take process. And companies need strict rules against things like
harassment, marginalizing colleagues, leaking company information, or insider trading. But we
work hard to keep rules at Netflix to a minimum and ensure any process is good (simple,
efficient, impactful). Our vacation policy, for example, is two words: "Take vacation." And our
expenses policy is just five words: "Act in Netflix's best interests." This (almost) no rules rule
gives employees the freedom to exercise their judgment. It also prevents the process creep
that typically happens when companies grow and try to dummy proof their organizations —
stifling creativity and making it harder for businesses to adapt.
You might think that this kind of freedom would lead to chaos. In reality, while we've had our
fair share of failures — and a few people have taken advantage of our culture in bad ways —
our emphasis on individual autonomy has created an extremely successful business, with
many opportunities for employees to develop and grow. In entertainment and technology, our
biggest threat is a lack of creativity, adaptability and innovation. It's why trying to minimize
rules and processes (rather than errors) — while giving people the freedom to use their own
judgment and learn from their mistakes — is a far superior recipe for long-term success.
UNCOMFORTABLY EXCITING
Netflix is programming for well over half a billion people globally — something no other
entertainment company has ever done before. Success requires us to be bold and ambitious,
to think differently, experiment and adapt (often quickly). This is true whether we're designing
a new product feature, working to improve our recommendations, developing a marketing
campaign or creating a TV show or game. Many people will be happier at companies that are
more stable or take fewer risks. Netflix works best if you value experimentation, enjoy the
uncomfortable excitement of a new or challenging project and have the resilience to thrive in
this environment.
ARTISTIC EXPRESSION
Representation matters. Our members come from many different backgrounds and cultures,
and they want to see a wide variety of stories and people on screen. This diversity is
wonderful and it can create real tension since we all have such different views about what's
acceptable — and what's harmful — on TV. While every show, film or game is different, we
approach them with the same set of principles: we support the artistic expression of the
creators we choose to work with; we program for a wide variety of audiences, cultures and
tastes; and we provide ratings, content advisories and parental controls in multiple languages
to help members choose what to watch or play. As employees, we support these principles,
even if some stories run counter to our personal values. And we understand that, depending
on our roles, we may need to work on TV shows, films or games we perceive to be harmful. If
you'd find it hard to support the breadth of our slate, Netflix is probably not the best place for
you.
GREAT AND ALWAYS BETTER
Netflix has come a long way since we mailed our first DVD in 1998. But we're nowhere close
to where we want to be in the future. It's why we care so much about the Dream Team, putting
people over process and creating an environment where everyone feels a sense of
responsibility to make us better. We believe this approach is the surest path to excellence and
long term success.
It's also why we constantly seek to improve our culture, not preserve it. Every new employee
helps shape how we work — finding new ways to accomplish more together. This creates a
better experience for our members, employees, creators and partners, which in turn propels
our long term growth and success. It's how we entertain the world and build a wildly
successful business.
"""

# ---------------------------------------------------------------------------
# Document https://jobs.netflix.com/work-life-philosophy
# ---------------------------------------------------------------------------
document_work_life_philosophy = """
A
 
supportive
 
supporting
 
cast
In our welcoming, collaborative, and supportive culture, you can be your best self.
Netflix 807


For Netflix, a big part of our focus on excellence is hiring people who are great at what they do and creating an environment where talent can thrive. Being part of our Dream Team means you’ll have a choice of offerings to assist you in the moments that matter. Because when you feel covered in life, you can focus on doing your best work. Everyone’s needs are unique and we want to empower you to take care of yourself and make decisions that are right for your family. That starts with a flexible approach to the support Netflix provides at work and beyond.
Netflix Image 3060
Here to help you get there
Parental Leave
We recognize that one of the most special events in an individual's life is welcoming a new child. Our parental leave policy is: "take care of your child and yourself." We encourage employees to think about the time they’ll need as a parent and have a conversation with their manager and talent partners to determine what's best for them and Netflix.

TIME AWAY
Our vacation policy is “take vacation”, and we actually do. Frankly, we intermix work and personal time quite a bit. Time away works differently at Netflix. We don’t have a prescribed 9-to-5 workday, so we don’t have prescribed time off policies for salaried employees, either. We don’t set a holiday and vacation schedule, so you can observe what’s important to you—including when your mind and body need a break. While time away may be observed differently depending on your location and role, we believe in taking the time you need so you are bringing your best to work.

Family Forming & Reproductive Health
Netflix offers a global family forming benefit to support employees looking for preservation, fertility, surrogacy, and/or adoption options. These benefits are available to employees and their spouse/domestic partner, regardless of marital status, gender, or sexual orientation.

Mental Health Benefits
At Netflix, we know you want to produce work you are proud of. In order to perform your best, you have to feel your best. Mental health is important to your overall health which is why Netflix offers various programs to support you and your dependents. Globally, we provide access to mindfulness and meditation as well as free counseling and coaching sessions.

Health Benefits
Medical benefits work differently based on the country you live in. So you may have medical coverage offered through Netflix only, a supplemental plan through Netflix that compliments your local insurance scheme, or you may have a monthly allowance to purchase benefits on your own. No matter what the case is, we make sure that you’re covered.

FINANCES THAT MATTER TO YOU
Personal Top Of Market
To help us attract and retain stunning colleagues, we pay employees at the top of their personal market. We do not think of these as “raises” and there is no raise pool to divide up. The market for talent is what it is and is not defined by set bands and grades. If your market adjusts, we do not have to wait for an annual compensation event to make changes. We hope through this compensation approach, we can dismantle pay disparities across gender and race.

Stock Option Program
Employees choose each year how much of their eligible compensation they want in salary versus stock options which provides the opportunity to benefit from Netflix’s long term success as our stock price appreciates. You can choose all cash, all options, or whatever combination suits you. These 10-year stock options are fully-vested and you keep them even if you leave Netflix.

Employee Giving Program
At Netflix, you have the ability to impact the world through your work, sharing great storytelling globally. We realize you may also want to make an impact in a more personal way by giving to or volunteering with charitable organizations you care about. We want to do our part to support you and the organizations important to you by matching your monetary donations and volunteer time through our Employee Giving Program.

Netflix Image 5004
03 NOV 17-NOV (©Tiffany Luong 2024)-00671 copy f p (1)
Convenience to Do Your Best Work
Expenses and Work-Related Travel
Our policy for travel, entertainment, gifts, and other expenses is 5 words long: “act in Netflix’s best interest.” We do not have a set travel policy at Netflix. Using our guiding principles of context, not control - you can use judgment to make decisions that are effective for the business and set you up for success. If in doubt, seek to understand.

Work, Not Drive
This program gives you flexibility to take a call, respond to your emails, and focus on work from a rideshare car on your way to and from the office.

Relocation Benefits
We are not bound by policies and restrictions of where great talent is located. When relocating at the request of Netflix to one of our offices, from one city, or country, to another, we have a team dedicated to supporting you and your family. We understand that relocating can be multi-faceted and that there is a lot to consider. Our goal is to remove distractions by providing inclusive, fair, and meaningful support, which allows our stunning colleagues to settle quickly and focus on their role at Netflix.
"""

# ---------------------------------------------------------------------------
# Document https://s22.q4cdn.com/959853165/files/doc_financials/2025/ar/99482238-46b2-4d0d-b292-40e6781bdf03.pdf
# ---------------------------------------------------------------------------
document_2025_annual_report_intro = """
PART I
Forward-Looking Statements
This Annual Report on Form 10-K contains forward-looking statements within the meaning of the federal securities laws. These forward-looking
statements include, but are not limited to, statements regarding: our core strategy; our ability to improve our content offerings and service; our future financial
performance, including expectations regarding revenues, deferred revenue, operating income and margin, net income, expenses, and profitability; liquidity,
including the sufficiency of our capital resources, cash requirements; net cash provided by (used in) operating activities, access to financing sources, and free
cash flows; capital allocation strategies, including any stock repurchases or repurchase programs; stock price volatility; impact of foreign exchange rate
fluctuations; expectations regarding hedging activity; impact of interest rate fluctuations; adequacy of existing facilities; future regulatory changes and their
impact on our business; intellectual property; cybersecurity; price changes and testing; artificial intelligence (“AI”); accounting treatment for changes related
to content assets; acquisitions; actions by competitors; membership growth, including impact of content and pricing changes on membership growth;
partnerships; advertising; multi-household usage; member viewing patterns; dividends; future contractual obligations, including unknown content obligations
and timing of payments; our global content and marketing investments, including investments in original programming, consumer products and live
experiences; impact of work stoppages; content amortization; resolutions of tax examinations; tax expense; unrecognized tax benefits; deferred tax assets; tax
deposits; resolutions of disputes and other proceedings; our ability to effectively manage change and growth; our company culture; expectations regarding the
transaction with Warner Bros. Discovery, Inc. ("WBD"); and our ability to attract and retain qualified employees and key personnel. These forward-looking
statements are subject to risks and uncertainties that could cause actual results and events to differ. A detailed discussion of these and other risks and
uncertainties that could cause actual results and events to differ materially from such forward-looking statements is included throughout this filing and
particularly in Item 1A: “Risk Factors” section set forth in this Annual Report on Form 10-K. All forward-looking statements included in this document are
based on information available to us on the date hereof, and we assume no obligation to revise or publicly release any revision to any such forward-looking
statement, except as may otherwise be required by law.
Item 1.
Business
ABOUT US
Netflix, Inc. (“Netflix”, the “Company”, “registrant”, “we”, or “us”) is one of the world’s leading entertainment services offering TV series, films, games
and live programming across a wide variety of genres and languages. Members can play, pause and resume watching as much as they want, anytime, anywhere,
and can change their plans at any time.
Our core strategy is to grow our business globally within the parameters of our operating margin target. We strive to continuously improve our members'
experience by offering compelling content that delights them and attracts new members. We aim to offer a range of pricing plans, including our ad-supported
subscription plan, to meet a variety of consumer needs. We seek to drive conversation around our content to further enhance member joy, and we are
continuously enhancing our user interface to help our members more easily choose content that they will find enjoyable.
BUSINESS SEGMENTS
We operate as one operating segment. Our revenues are primarily derived from monthly membership fees for services related to streaming content to our
members. See Note 13, Segment and Geographic Information, in the accompanying notes to our consolidated financial statements for further detail.
COMPETITION
The market for entertainment video is intensely competitive and subject to rapid change. We compete with a broad set of activities for consumers’ leisure
time, including other entertainment video providers, such as linear television, streaming entertainment providers (including those that provide pirated content),
video gaming providers, open content platform providers, which provide access to user-generated and professionally produced content, as well as more broadly
against other sources of entertainment, such as social media, that our members could choose in their moments of free time. We also compete against
entertainment video providers and content producers in obtaining content for our service, both for licensed content and for original content projects.
While consumers may maintain simultaneous relationships with multiple entertainment sources, we strive for consumers to choose us in their moments of
free time. We have often referred to this choice as our objective of “winning moments of truth.” In attempting to win these moments of truth with our members,
we seek to continually improve our service, including both our technology and our content offerings.
1
Table of Contents
INTELLECTUAL PROPERTY
We regard our trademarks, service marks, copyrights, patents, domain names, trade dress, trade secrets, proprietary technologies and similar intellectual
property as important to our success. We use a combination of patent, trademark, copyright and trade secret laws and confidentiality agreements to protect our
proprietary intellectual property. Our intellectual property rights extend to our technology, business processes, the content we produce and distribute through
our service, and the consumer products and experiences based thereon. We use the intellectual property of third parties in creating some of our content,
merchandising our products and marketing our service. Our ability to provide our members with content they can watch depends on studios, content providers
and other rights holders licensing rights, including distribution rights, to such content and certain related elements thereof, such as the public performance of
music contained within the content we distribute. The license periods and the terms and conditions of such licenses vary. Our ability to protect and enforce our
intellectual property rights is subject to certain risks and from time to time we encounter disputes over rights and obligations concerning intellectual property.
We cannot provide assurance that we will prevail in any intellectual property disputes.
REGULATION
The media landscape and the internet delivery of content have seen growing regulatory action. Historically, media has been highly regulated in many
countries. We are seeing some of these legacy regulatory frameworks be updated and expanded to address services like ours. In particular, we are seeing some
countries update their cultural support legislation to include services like Netflix. This includes investment obligations, levies, and content catalog quotas.
Some even restrict the extent of ownership rights we can have both in our service and in our content. In certain countries, regulators are also looking at
restrictions that could require formal reviews of and/or adjustments to content that appears on our service in their country. In general these regulations impact
all services and may make operating in certain jurisdictions more expensive or restrictive as to the content offerings we may provide.
HUMAN CAPITAL
Our business is to entertain the world across different countries, cultures, languages and tastes. To entertain an audience this global, our Company needs
to reflect the world and the variety of stories we tell. To help ensure our workforce is representative of the members we serve, we employ people in multiple
countries around the world and work to maintain a global culture of inclusion. We view our employees and our culture as key to our success. As of
December 31, 2025, we had approximately 16,000 full-time employees. Of these, approximately 10,900 (68%) were located in the United States and Canada,
2,500 (16%) in Europe, Middle East, and Africa, 1,900 (12%) in Asia-Pacific and 700 (4%) in Latin America. We also have a number of employees engaged in
content production, some of whom are part-time or temporary, and whose numbers fluctuate throughout the year and may be covered by collective bargaining
agreements.
We believe an important component of our success is our company culture as detailed in the “Netflix Culture Memo”, which was updated in 2024. Our
culture is focused on excellence and creating an environment where talented people can thrive — lifting ourselves, each other and our audiences higher and
higher. We engage employees and seek feedback through regular town halls, surveys, business reviews and memos, which we often share broadly, inviting
comments. We aim to attract and retain great people — representing a broad array of perspectives and skills — to work together as a dream team. For more
people and cultures to see themselves reflected on screen, it is important that our employee base represents the communities we serve.
We aim generally to pay our employees at their personal top of market, and they generally are able to choose the form of their compensation between
cash and stock options. This permits employee compensation to be highly personalized and reflective of each employee's individual needs and preferences. We
conduct pay equity analyses at least annually, and have adopted practices to help ensure that employees from underrepresented groups are not being underpaid
based on gender identity (globally) and race or ethnicity (United States (“U.S.”)) relative to others doing the same or similar work under comparable
circumstances. We aim to rectify any pay gaps that we find through this analysis.
We care about the health and well-being of our employees and their families and provide a variety of benefit programs based on region, including health
benefits. In the U.S., employees generally receive an annual cash health benefit allowance that they may allocate to medical, dental and vision premiums in a
way that makes sense for them. Employees have access to a host of other benefits, including mental health, childcare, family planning and a company match
for charitable donations.
We believe that our approach to human capital resources has been instrumental in our growth, and has made Netflix a desirable destination for employees.
OTHER INFORMATION
We maintain a website at www.netflix.com. The contents of our website are not incorporated in, or otherwise to be regarded as part of, this Annual Report
on Form 10-K. We make available, free of charge on our website, access to our Annual Report on Form 10-K, our Quarterly Reports on Form 10-Q, our
Current Reports on Form 8-K and amendments to those reports filed or furnished pursuant to Section 13(a) or 15(d) of the Securities Exchange Act of 1934, as
amended (the “Exchange Act”), as soon as reasonably practicable after we file or furnish them electronically with the Securities and Exchange Commission
(“SEC”).
2
Table of Contents
Investors and others should note that we announce material financial and other information to our investors using our investor relations website
(ir.netflix.net), SEC filings, press releases, public conference calls and webcasts. We use these channels as well as social media and blogs to communicate with
our members and the public about our company, our services and other issues. It is possible that the information we post on social media and blogs could be
deemed to be material information. Therefore, we encourage investors, the media, and others interested in our company to review the information we post on
the social media channels and blogs listed on our investor relations website."""

# ---------------------------------------------------------------------------
# Chunking function
# ---------------------------------------------------------------------------

def chunk_document(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP, min_chunk_size=MIN_CHUNK_SIZE):
    """
    Splits a document into overlapping chunks that respect natural text boundaries.

    Priority order for where a chunk ends:
      1. Paragraph boundary (double newline)
      2. Bulleted-list boundary
      3. Sentence boundary (. ! ? followed by whitespace)
      4. Word boundary (any whitespace)

    Words are never split. Chunks are at least min_chunk_size characters unless the
    remaining text is shorter. Consecutive chunks share `overlap` characters of context.
    """
    def is_list_line(line: str) -> bool:
        stripped = line.lstrip()
        return (
            stripped.startswith("- ")
            or stripped.startswith("* ")
            or stripped.startswith("• ")
            or stripped.startswith("● ")
            or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in {". ", ") "})
        )
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        # ── Calculate where this chunk would ideally end ──────────────────────
        # Aim for a chunk of approximately chunk_size characters.
        target_end = start + chunk_size

        # If the target end reaches or passes the end of the document,
        # just take whatever text is left and we're done.
        if target_end >= text_length:
            remaining = text[start:].strip()  # Strip surrounding whitespace
            if remaining:                     # Skip if only whitespace remains
                chunks.append(remaining)
            break                             # No more text to process

        # ── Define the earliest acceptable end position ───────────────────────
        # We never end a chunk before this point so every chunk is at least
        # min_chunk_size characters long (barring the final leftover).
        search_floor = start + min_chunk_size
        end = -1

        # 1: Paragraph boundary (\n\n) ────────────────────────────
        # Search backwards from target_end so we land as close to the target
        # size as possible while still ending at a blank line.
        para_pos = text.rfind('\n\n', search_floor, target_end + 1)

        if para_pos != -1:
            # End the chunk right after the double newline so the newlines
            # belong to the current chunk, not the next one.
            end = para_pos + 2
            
        # 2. Bulleted-list boundary
        if end == -1:
            # Look for a newline where the NEXT line starts a bullet/list item.
            # End before the list starts, so the whole list stays together more often.
            for i in range(target_end, search_floor - 1, -1):
                if text[i] == "\n":
                    next_line_start = i + 1
                    next_line_end = text.find("\n", next_line_start)
                    if next_line_end == -1:
                        next_line_end = text_length
                    next_line = text[next_line_start:next_line_end]

                    if is_list_line(next_line):
                        end = i
                        break

            # Also look for the end of a bulleted-list block:
            # newline where current line is a bullet and next line is not.
            if end == -1:
                for i in range(target_end, search_floor - 1, -1):
                    if text[i] == "\n":
                        prev_line_start = text.rfind("\n", 0, i)
                        prev_line_start = 0 if prev_line_start == -1 else prev_line_start + 1
                        prev_line = text[prev_line_start:i]

                        next_line_start = i + 1
                        next_line_end = text.find("\n", next_line_start)
                        if next_line_end == -1:
                            next_line_end = text_length
                        next_line = text[next_line_start:next_line_end]

                        if is_list_line(prev_line) and not is_list_line(next_line):
                            end = i
                            break

        # 3: Sentence boundary (. ! ?) ────────────────────────
        # Walk backwards from target_end looking for sentence-ending
        # punctuation immediately followed by whitespace (space, newline, tab).
        if end == -1:
            for i in range(min(target_end, text_length - 1), search_floor - 1, -1):
                if text[i] in '.!?' and (i + 1 < text_length) and text[i + 1] in ' \n\t\r':
                    # Include the punctuation mark itself in the current chunk.
                    end = i + 1
                    break

            # 4: Word boundary (whitespace) ────────────────────
            # Walk backwards from target_end to find any whitespace character
            # so we never slice through the middle of a word.
        if end == -1:
            for i in range(min(target_end, text_length - 1), search_floor - 1, -1):
                if text[i] in ' \n\t\r':
                    end = i
                    break

        if end == -1:
            # No boundary of any kind was found in the search window.
            # This should only happen with pathologically long single
            # words; fall back to a hard cut at the target position.
            end = min(target_end, text_length)

        # ── List-integrity guard ──────────────────────────────────────────────
        # Bullets are often multi-line (the continuation doesn't start with ●).
        # Instead of checking only the immediate last line, scan backward from
        # `end` to see if ANY bullet appears before a blank line — that means
        # we're inside a list block and should extend forward to include it all.
        if end < text_length:
            def is_inside_list_block(pos):
                """Return True if a bullet line exists between `start` and `pos`
                with no blank line separating it from `pos`."""
                scan = pos
                while scan > start:
                    nl = text.rfind("\n", start, scan)
                    if nl == -1:
                        return is_list_line(text[start:scan].strip())
                    line = text[nl + 1:scan].strip()
                    if not line:          # blank line — outside any list block
                        return False
                    if is_list_line(line):
                        return True
                    scan = nl             # check the previous line
                return False

            if is_inside_list_block(end):
                hard_limit = min(start + chunk_size * 2, text_length)
                scan = end
                while scan < text_length and text[scan] in "\n\r":
                    scan += 1

                while scan < hard_limit:
                    nl = text.find("\n", scan)
                    line_end = nl if nl != -1 else text_length
                    line = text[scan:line_end].strip()

                    if not line:          # blank line — list definitely ended
                        break

                    if is_list_line(line):
                        # Bullet line — include it and keep going
                        end = (nl + 1) if nl != -1 else text_length
                        if nl == -1:
                            break
                        scan = nl + 1
                    else:
                        # Non-bullet line: could be a continuation or post-list prose.
                        # Look ahead up to 3 non-empty lines for another bullet.
                        # If found → this is a continuation; include it and keep going.
                        # If not  → we've cleared the list; include this last line and stop.
                        la = (nl + 1) if nl != -1 else text_length
                        found_bullet = False
                        steps = 0
                        while steps < 3 and la < text_length:
                            la_nl = text.find("\n", la)
                            la_end = la_nl if la_nl != -1 else text_length
                            la_line = text[la:la_end].strip()
                            if la_line:
                                if is_list_line(la_line):
                                    found_bullet = True
                                    break
                                steps += 1
                            la = (la_nl + 1) if la_nl != -1 else text_length

                        # Always include the current line (continuation or final line)
                        end = (nl + 1) if nl != -1 else text_length
                        if not found_bullet:
                            break        # No bullets ahead — full list captured
                        if nl == -1:
                            break
                        scan = nl + 1

        # ── Store the completed chunk ─────────────────────────────────────────
        chunk = text[start:end].strip()  # Remove leading/trailing whitespace
        if chunk:                        # Discard chunks that are only whitespace
            chunks.append(chunk)

        # ── Advance the start position for the next chunk ────────────────────
        # Subtract `overlap` so the next chunk re-reads the last `overlap`
        # characters of this chunk, giving the model continuity across boundaries.
        next_start = end - overlap

        # Safety guard: if subtracting overlap would not move us forward
        # (e.g. the chunk was very short), skip the overlap to avoid looping.
        if next_start <= start:
            next_start = end

        start = next_start  # Move to the start of the next chunk

    return chunks

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_documents():
    return [
        {"text": document_culture_memo, "source": "Netflix Culture Memo"},
        {"text": document_work_life_philosophy, "source": "Netflix Work-Life Philosophy"},
        {"text": document_2025_annual_report_intro, "source": "Netflix 2025 Annual Report Introduction"},
    ]


def prepare_chunks(documents):
    chunks = []
    ids = []
    metadatas = []

    for doc in documents:
        if not doc["text"].strip():
            continue

        doc_chunks = chunk_document(doc["text"])
        doc_ids = [str(uuid.uuid4()) for _ in range(len(doc_chunks))]
        doc_metadatas = [
            {"source": doc["source"], "chunk_index": i}
            for i in range(len(doc_chunks))
        ]

        chunks.extend(doc_chunks)
        ids.extend(doc_ids)
        metadatas.extend(doc_metadatas)

    return chunks, ids, metadatas

# ---------------------------------------------------------------------------
# Build vector index (in-memory) at startup
# ---------------------------------------------------------------------------
def build_vector_store():
    global collection

    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing. Cannot build vector store.")

    documents = load_documents()
    chunks, ids, metadatas = prepare_chunks(documents)

    if not chunks:
        raise ValueError("No document text was provided. Add source text before launching the app.")

    print("Building vector index...")

    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    embeddings = [item.embedding for item in embedding_response.data]

    chroma_client = chromadb.Client()  # in-memory
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"Index ready — {len(chunks)} chunks loaded.")

# ---------------------------------------------------------------------------
# Retrieval + formatting
# ---------------------------------------------------------------------------
def retrieve_context(message):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing. Cannot process queries.")

    if collection is None:
        raise RuntimeError("Vector store is not initialized.")

    query_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[message],
    )
    query_embedding = query_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "", []

    context_parts = []
    citations = []

    for doc_text, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown Source")
        chunk_index = metadata.get("chunk_index", "?")

        citations.append((source, chunk_index))
        context_parts.append(
            f"Source: {source} | Chunk: {chunk_index}\n{doc_text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    unique_citations = list(dict.fromkeys(citations))

    return context, unique_citations


def format_sources_for_output(citations):
    if not citations:
        return ""

    lines = ["\n\nSources used:"]
    for source, chunk_index in citations:
        lines.append(f"- {source} | Chunk: {chunk_index}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Chat response function
# ---------------------------------------------------------------------------
def respond_ai(message, history):
    try:
        if not message or not message.strip():
            return "Please enter a question about Netflix culture, benefits, or company information."

        context, sources = retrieve_context(message)

        if not context:
            return "I do not know based on the provided source materials."

        system_message_enhanced = f"{SYSTEM_MESSAGE}\n\nContext:\n{context}"

        messages = [{"role": "system", "content": system_message_enhanced}]

        # Gradio ChatInterface may pass history as message dicts or tuples depending on setup.
        # This block safely normalizes common formats.
        if history:
            for item in history:
                if isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    user_msg, assistant_msg = item
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": message})

        chat_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )

        answer = chat_response.choices[0].message.content.strip()
        #sources_block = format_sources_for_output(sources)

        # return answer + sources_block
        return answer

    except RuntimeError as e:
        return f"Configuration error: {e}"
    except ValueError as e:
        return f"Setup error: {e}"
    except Exception as e:
        print(f"Unexpected error in respond_ai: {e}")
        return "Sorry — I ran into an error while generating a response. Please try again."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_demo():
    return gr.ChatInterface(
        fn=respond_ai,
        title="Netflix Culture Chatbot",
        description=(
            "Ask questions about Netflix culture, benefits, and company information. "
            "This is a personal demo project using RAG (Retrieval-Augmented Generation). "
            "Answers are limited to the provided source materials."
        ),
        examples=[
            "What does Netflix mean by the Dream Team?",
            "How does Netflix describe its vacation policy?",
            "What does Netflix say about compensation?",
            "What mental health support is mentioned?",
            "How does Netflix describe people over process?",
        ],
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        build_vector_store()
    except Exception as e:
        print(f"Startup error: {e}")

    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
