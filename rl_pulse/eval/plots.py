# define functions to save plots periodically

def makeParamDiffPlots(paramDiff, prefix):
    if len(paramDiff) < 2:
        return
    diffEps = [_[0] for _ in paramDiff]
    actorDiffs = np.array([_[1] for _ in paramDiff])
    criticDiffs = np.array([_[2] for _ in paramDiff])
    
    numFigs = 0
    for d in range(np.shape(actorDiffs)[1]):
        plt.plot(diffEps, actorDiffs[:,d], label=f"parameter {d}")
        if ((d+1) % 6 == 0):
            # 10 lines have been added to plot, save and start again
            plt.title(f"Actor parameter MSE vs target networks (#{numFigs})")
            plt.xlabel('Generation number')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.legend()
            # plt.gcf().set_size_inches(12,8)
            plt.savefig("../data/" + prefix + \
                f"/actor_param_MSE{numFigs:02}.png", bbox_inches='tight')
            plt.clf()
            numFigs += 1
    plt.title(f"Actor parameter MSE vs target networks (#{numFigs})")
    plt.xlabel('Generation number')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    # plt.gcf().set_size_inches(12,8)
    plt.savefig("../data/" + prefix + f"/actor_param_MSE{numFigs:02}.png", \
        bbox_inches='tight')
    plt.clf()

    numFigs = 0
    for d in range(np.shape(criticDiffs)[1]):
        plt.plot(diffEps, criticDiffs[:,d], label=f"parameter {d}")
        if ((d+1) % 6 == 0):
            # 10 lines have been added to plot, save and start again
            plt.title(f"Critic parameter MSE vs target networks (#{numFigs})")
            plt.xlabel('Generation number')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.legend()
            # plt.gcf().set_size_inches(12,8)
            plt.savefig("../data/" + prefix + \
                f"/critic_param_MSE{numFigs:02}.png", bbox_inches='tight')
            plt.clf()
            numFigs += 1
    plt.title(f"Critic parameter MSE vs target networks (#{numFigs})")
    plt.xlabel('Generation number')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    # plt.gcf().set_size_inches(12,8)
    plt.savefig("../data/" + prefix + f"/critic_param_MSE{numFigs:02}.png", \
        bbox_inches='tight')
    plt.clf()

def makePopFitPlot(popFitnesses, prefix):
    """Make a plot of population fitnesses
    
    Arguments:
        popFitnesses: A list of tuples (generation number, [array of fitnesses])
        prefix: The prefix to add to the plot files
    """
    fig = px.scatter(popFitnesses, x='generation', y='fitness', \
        color='individual', symbol='mutatedRecently', size='fitnessInd')
    fig.update_layout(title={'text': 'Population fitness vs generation', \
            'x': .4, 'xanchor': 'center'}, \
        xaxis_title='Generation', yaxis_title='Fitness')
    fig.write_image("../data/" + prefix + f"/pop_fit.png", \
        width=900, height=500, scale=2)

def makeTestPlot(testMat, prefix):
    fig = px.scatter(testMat, x='generation', y='fitness', \
        symbol='type', size='fitnessInd')
    fig.update_layout(title={'text': 'Test fitness vs generation',\
            'x': .4, 'xanchor': 'center'}, \
        xaxis_title='Generation', yaxis_title='Fitness')
    fig.write_image("../data/" + prefix + f"/test_fit.png", \
        width=900, height=500, scale=2)
