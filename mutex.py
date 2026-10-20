"""mutex.py: try every order in which two threads' steps can run, and report what can go wrong.

Run it from a terminal in the folder that holds this file:
    python mutex.py           (Windows)   or   python3 mutex.py   (macOS / Linux)

Your work is marked with TODO. Everything else already works: run the file before you change
anything; each experiment that reaches an unwritten TODO says which one and the rest still run.
Nothing here is random and nothing depends on timing. The program lists EVERY interleaving of the
two threads, so the output is the same on every computer and every run.
"""

from collections import deque  # a queue with fast removal from the front: the to-do list of the search

INCREMENTS = 3  # Experiment 1: how many times each thread adds 1 to count
VISITS = 2  # Experiment 2: how many times each thread passes through its critical section
SIZE = 1  # B-d: test whether a one-slot buffer still preserves safety and progress
ITEMS = 3  # Experiment 3: how many items the producer makes (and the consumer takes)


class NotWrittenYet(Exception):  # a custom error the starter raises wherever a TODO has not been written yet
    pass  # it needs no extra behavior; its name and message say everything


# ---------------------------------------------------------------------------
# How a thread is written down
#   A thread's program is a list of steps, numbered 0, 1, 2, ... Each step is a pair (text, action).
#   The action is a function action(sh, me). sh is a dictionary holding the shared variables (the
#   action may change it); me is this thread's number, 0 or 1, so the other thread is 1 - me.
#   The action returns the number of the step this thread runs next, or None when the step cannot
#   happen yet (the thread is blocked; the action must then leave sh unchanged).
#   A thread whose next step number equals the length of its program has finished.
#   A step whose text starts with "CS:" is inside the critical section.
#   One step is one atomic action: the other thread can run between two steps, never inside one.
# ---------------------------------------------------------------------------
def put(sh, name, value, nxt):  # a helper for actions that assign: sh[name] = value, then go to step nxt
    sh[name] = value  # the assignment
    return nxt  # the number of the step that runs next


def again(sh, me, first, done):  # the last step of a loop: count one pass, then go round again or finish
    sh[f"left{me}"] -= 1  # one pass fewer left for this thread (left0 belongs to thread 0, left1 to thread 1)
    return first if sh[f"left{me}"] > 0 else done  # back to step `first` while passes remain, else to step `done`


def sem_wait(name, nxt):  # builds a semWait step on the semaphore called `name`, followed by step nxt
    def action(sh, me):  # what the step does when the thread tries to take it
        if sh[name] == 0:  # no unit is available...
            return None  # ...so the thread is blocked here, and nothing changes
        sh[name] -= 1  # take one unit
        return nxt  # and carry on
    return (f"semWait({name})", action)  # the step: its text and its action


def sem_signal(name, nxt):  # builds a semSignal step on the semaphore called `name`, followed by step nxt
    return (f"semSignal({name})", lambda sh, me: put(sh, name, sh[name] + 1, nxt))  # give one unit back; never blocks


def append_item(nxt):  # builds the producer's critical-section step: one more item in the buffer
    return ("CS: append an item", lambda sh, me: put(sh, "items", sh["items"] + 1, nxt))  # items counts what the buffer holds


def take_item(nxt):  # builds the consumer's critical-section step: one item fewer in the buffer
    return ("CS: take an item", lambda sh, me: put(sh, "items", sh["items"] - 1, nxt))  # the item leaves the buffer


# ---------------------------------------------------------------------------
# The checker: a breadth-first search over every state the two threads can reach
#   A state = (each thread's next step number, the values of every shared variable).
# ---------------------------------------------------------------------------
def freeze(sh):  # turns the shared dictionary into a tuple, because only unchangeable values can go in a set
    return tuple(sorted(sh.items()))  # e.g. (("flag0", True), ("flag1", False), ("turn", 1))


def explore(threads, init):  # visits every reachable state; returns how each was first reached, and every move
    start = ((0,) * len(threads), freeze(init))  # both threads at step 0, shared variables at their starting values
    came_from = {start: None}  # for each state seen: (the state before it, the thread that moved); None for the start
    moves = {}  # for each state: the list of states one step away
    queue = deque([start])  # states whose moves have not been tried yet
    while queue:  # until every reachable state has been expanded
        state = queue.popleft()  # oldest first, so every state is first reached by a shortest path
        pcs, frozen = state  # the step numbers and the shared variables
        moves[state] = []  # filled in by the loop below
        for t, program in enumerate(threads):  # let thread 0, then thread 1, try its next step
            if pcs[t] == len(program):  # this thread has finished...
                continue  # ...so it has no step to take
            sh = dict(frozen)  # a fresh, changeable copy of the shared variables
            nxt = program[pcs[t]][1](sh, t)  # run thread t's next step on that copy
            if nxt is None:  # the step is blocked...
                continue  # ...so thread t cannot move in this state
            new = (pcs[:t] + (nxt,) + pcs[t + 1:], freeze(sh))  # the state after the step
            moves[state].append(new)  # remember the move
            if new not in came_from:  # a state never seen before...
                came_from[new] = (state, t)  # ...remember how we got there
                queue.append(new)  # ...and expand it later
    return came_from, moves  # the keys of came_from are all the reachable states, in the order they were found


def trace(came_from, threads, end):  # the moves that lead from the start to state `end`
    steps = []  # collected backwards, from `end` to the start
    while came_from[end] is not None:  # only the start has no previous state
        before, t = came_from[end]  # the state before, and the thread that moved
        steps.append((t, threads[t][before[0][t]][0], end[1]))  # (thread, text of the step it took, shared variables after)
        end = before  # one step further back
    return steps[::-1]  # reversed, so the first move comes first


def show(steps):  # prints a trace, one line per step
    for t, text, frozen in steps:  # every move in order
        shared = "  ".join(f"{k}={v}" for k, v in frozen if not k.startswith(("left", "reg")))  # the shared variables, minus loop counters and registers
        print(f"      T{t}  {text:<42} {shared}")  # e.g. "T0  flag[me] = True    flag0=True  flag1=False"


def status(threads, state):  # describes what each thread is doing in a state, e.g. "T0 blocked at semWait(e)"
    words = []  # one description per thread
    for t, program in enumerate(threads):  # each thread
        pc = state[0][t]  # its next step number
        if pc == len(program):  # past its last step
            words.append(f"T{t} has finished")  # nothing left to do
            continue  # next thread
        sh = dict(state[1])  # a copy of the shared variables to try the step on
        nxt = program[pc][1](sh, t)  # try the step
        if nxt is None:  # the step cannot happen
            how = "is blocked at"  # waiting inside the OS, using no processor time
        elif nxt == pc and freeze(sh) == state[1]:  # the step leads straight back to itself and changes nothing
            how = "spins forever on"  # busy waiting that can never end
        else:  # the step can still change something
            how = "can still take"  # (only possible in a stuck state that loops through several states)
        words.append(f"T{t} {how} '{program[pc][0]}'")  # e.g. "T1 spins forever on 'while turn != me: wait'"
    return "; ".join(words)  # both descriptions on one line


def in_cs(program, pc):  # is a thread whose next step is pc inside its critical section?
    return pc < len(program) and program[pc][0].startswith("CS:")  # only steps marked "CS:" count


def check(name, model):  # explores one model and prints what can and cannot happen
    threads, init = model["threads"], model["init"]  # the two programs, and the shared variables' starting values
    came_from, moves = explore(threads, init)  # every reachable state
    states = list(came_from)  # in breadth-first order: a state earlier in the list is fewer steps from the start
    print(f"  {name}: {len(states)} reachable states")  # how big the search was
    done = [s for s in states if all(pc == len(p) for pc, p in zip(s[0], threads))]  # states where both threads have finished
    if "result" in model:  # Experiment 1 only asks which final values are possible
        values = sorted({dict(s[1])[model["result"]] for s in done})  # each distinct final value once, smallest first
        print(f"    possible final values of {model['result']}: {', '.join(map(str, values))}")  # the answer
        return  # nothing else to check
    if not all(any(text.startswith("CS:") for text, _ in p) for p in threads):  # some thread has no step marked "CS:"...
        print("    WARNING: a thread has no step marked 'CS:', so the mutual exclusion check means nothing")  # ...so the next check cannot see it inside
    both_in = [s for s in states if all(in_cs(p, pc) for p, pc in zip(threads, s[0]))]  # states with both threads inside
    if both_in:  # mutual exclusion can fail...
        print("    both threads in the critical section at once: POSSIBLE. The shortest way:")  # say so
        show(trace(came_from, threads, both_in[0]))  # and show how (the first one found is the closest to the start)
    else:  # no interleaving gets both inside
        print("    both threads in the critical section at once: never")  # mutual exclusion holds
    if "broken" in model:  # an extra test of the shared variables (Experiment 3: the buffer's limits)
        bad = [s for s in states if model["broken"](dict(s[1]))]  # states that fail the test
        if bad:  # the test can fail...
            print(f"    {model['broken'](dict(bad[0][1]))}: POSSIBLE. The shortest way:")  # say what went wrong
            show(trace(came_from, threads, bad[0]))  # and how
        else:  # no reachable state fails it
            print("    buffer overflow or underflow: never")  # the limits always hold
    behind = {}  # the moves turned around: for each state, the states that lead into it
    for state, outs in moves.items():  # every state...
        for out in outs:  # ...and every move out of it
            behind.setdefault(out, []).append(state)  # record the move backwards
    can_finish = set(done)  # states from which both threads can still finish; it grows backwards from the finished ones
    queue = deque(done)  # states whose predecessors have not been looked at yet
    while queue:  # walk the moves backwards
        state = queue.popleft()  # a state that can finish...
        for before in behind.get(state, []):  # ...so every state that leads into it can finish too
            if before not in can_finish:  # new
                can_finish.add(before)  # mark it
                queue.append(before)  # and walk back from it later
    stuck = [s for s in states if s not in can_finish]  # reachable states from which finishing is impossible
    if stuck:  # some interleaving traps the threads (deadlock, or spinning forever)
        traps = [s for s in stuck if all(out == s for out in moves[s])]  # stuck states where no step changes anything at all
        end = (traps or stuck)[0]  # show the nearest such trap (or, if there is none, the nearest stuck state)
        where = "a dead end (a state no step can change)" if traps else "a stuck state"  # what the trace below leads to
        print(f"    can always still finish: NO ({len(stuck)} of {len(states)} states are stuck). The shortest way to {where}:")  # say so
        show(trace(came_from, threads, end))  # the moves that lead there (none if the start itself is stuck)
        print(f"      stuck: {status(threads, end)}")  # what each thread is doing there, forever
    else:  # from every reachable state, some order of steps still finishes both threads
        print("    can always still finish: yes")  # no deadlock and no state that spins forever


# ---------------------------------------------------------------------------
# Experiment 1: count = count + 1 is really three steps
# ---------------------------------------------------------------------------
def increment_race():  # each thread adds 1 to count INCREMENTS times; registers are private, count is shared
    program = [  # the same program for both threads
        ("reg = count", lambda sh, me: put(sh, f"reg{me}", sh["count"], 1)),  # load: copy count into my own register
        ("reg = reg + 1", lambda sh, me: put(sh, f"reg{me}", sh[f"reg{me}"] + 1, 2)),  # add 1, in the register only
        ("count = reg", lambda sh, me: put(sh, "count", sh[f"reg{me}"], 3)),  # store: copy the register back into count
        ("repeat if increments remain", lambda sh, me: again(sh, me, 0, 4)),  # loop back to step 0, or finish
    ]  # reg0 and reg1 live in sh too, but each is only ever touched by its own thread
    init = {"count": 0, "reg0": 0, "reg1": 0, "left0": INCREMENTS, "left1": INCREMENTS}  # starting values
    return {"threads": [program, program], "init": init, "result": "count"}  # the model


# ---------------------------------------------------------------------------
# Experiment 2: four ways to guard a critical section
# ---------------------------------------------------------------------------
def look_then_flag():  # wait while the other thread's flag is up, then raise my own and go in
    program = [  # the same program for both threads
        ("while flag[other]: wait", lambda sh, me: 0 if sh[f"flag{1 - me}"] else 1),  # spin on step 0 while the other wants in
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 2)),  # raise my flag
        ("CS: critical section", lambda sh, me: 3),  # the critical section itself
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 4)),  # lower my flag on the way out
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 5)),  # loop back, or finish
    ]  # the look (step 0) and the flag (step 1) are two separate steps
    init = {"flag0": False, "flag1": False, "left0": VISITS, "left1": VISITS}  # nobody wants in yet
    return {"threads": [program, program], "init": init}  # the model


def flag_then_look():  # raise my flag first, then wait while the other thread's flag is up
    program = [  # the same program for both threads
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 1)),  # raise my flag
        ("while flag[other]: wait", lambda sh, me: 1 if sh[f"flag{1 - me}"] else 2),  # spin on step 1 while the other wants in
        ("CS: critical section", lambda sh, me: 3),  # the critical section itself
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 4)),  # lower my flag on the way out
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 5)),  # loop back, or finish
    ]  # the same two steps as above, in the opposite order
    init = {"flag0": False, "flag1": False, "left0": VISITS, "left1": VISITS}  # nobody wants in yet
    return {"threads": [program, program], "init": init}  # the model


def strict_turns():  # one shared turn variable; thread 0 visits once and leaves, thread 1 wants VISITS visits
    program = [  # the same program for both threads
        ("while turn != me: wait", lambda sh, me: 0 if sh["turn"] != me else 1),  # spin until it is my turn
        ("CS: critical section", lambda sh, me: 2),  # the critical section itself
        ("turn = other", lambda sh, me: put(sh, "turn", 1 - me, 3)),  # hand the turn to the other thread
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 4)),  # loop back, or finish
    ]  # the threads can only ever go in the order 0, 1, 0, 1, ...
    init = {"turn": 0, "left0": 1, "left1": VISITS}  # thread 0 has the first turn but needs only one visit
    return {"threads": [program, program], "init": init}  # the model


def peterson():  # Peterson's algorithm: raise my flag, give the turn away, wait only if the other wants in AND has the turn
    program = [
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 1)),
        ("turn = other", lambda sh, me: put(sh, "turn", 1 - me, 2)),
        ("while flag[other] and turn == other: wait",
         lambda sh, me: 2 if sh[f"flag{1 - me}"] and sh["turn"] == 1 - me else 3),
        ("CS: critical section", lambda sh, me: 4),
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 5)),
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    init = {"flag0": False, "flag1": False, "turn": 0,
            "left0": VISITS, "left1": VISITS}
    return {"threads": [program, program], "init": init}


LOCKS = [  # Experiment 2 checks these in order; an unwritten TODO stops the experiment there
    ("Look, then flag", look_then_flag),  # can both get in?
    ("Flag, then look", flag_then_look),  # can both get stuck?
    ("Strict turns", strict_turns),  # what if one thread stops coming?
    ("Peterson's algorithm", peterson),  # TODO 1: write peterson() above
]  # the end of the list


# ---------------------------------------------------------------------------
# Experiment 3: a bounded buffer guarded by three semaphores
#   s = 1 (one thread at a time in the buffer), n = 0 (items waiting), e = SIZE (empty slots).
#   In this checker a semaphore's value never goes below 0: semWait simply cannot happen while the value
#   is 0. (The guide's version lets the value go negative to count the waiters; which steps can happen
#   next is the same.)
# ---------------------------------------------------------------------------
def bounded_buffer():  # thread 0 is the producer, thread 1 the consumer; each handles ITEMS items
    producer = [
        sem_wait("e", 1), sem_wait("s", 2), append_item(3),
        sem_signal("s", 4), sem_signal("n", 5),
        ("repeat if items remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    consumer = [
        sem_wait("n", 1), sem_wait("s", 2), take_item(3),
        sem_signal("s", 4), sem_signal("e", 5),
        ("repeat if items remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    init = {"s": 1, "n": 0, "e": SIZE, "items": 0, "left0": ITEMS, "left1": ITEMS}  # an empty buffer
    broken = lambda sh: ("buffer overflow" if sh["items"] > SIZE else "buffer underflow" if sh["items"] < 0 else None)  # a broken state, or None if fine
    return {"threads": [producer, consumer], "init": init, "broken": broken}  # the model


def main():  # runs the three experiments in order
    print(f"Experiment 1: two threads each add 1 to count {INCREMENTS} times (count = count + 1 is load, add, store)")  # the title
    check("Increment race", increment_race())  # explore it
    print("\nExperiment 2: four ways to guard a critical section")  # the title
    try:  # one lock after another
        for name, build in LOCKS:  # in order
            model = build()  # an unwritten TODO stops here
            visits = f"visits: T0 {model['init']['left0']}, T1 {model['init']['left1']}"  # how often each thread enters, read from the model itself
            check(f"{name} ({visits})", model)  # explore it
    except NotWrittenYet as missing:  # a TODO was reached
        print(f"  -> stopped here: {missing}")  # say which one
    print(f"\nExperiment 3: a producer and a consumer pass {ITEMS} items through a buffer that holds {SIZE}")  # the title
    try:  # the semaphore solution
        check("Bounded buffer", bounded_buffer())  # an unwritten TODO stops here
    except NotWrittenYet as missing:  # a TODO was reached
        print(f"  -> stopped here: {missing}")  # say which one


if __name__ == "__main__":  # true when the file is run directly (not imported)
    main()  # start the program
