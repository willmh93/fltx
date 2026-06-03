#include <catch2/catch_config.hpp>
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{
    Catch::Session session;

    Catch::ConfigData config;
    config.runOrder = Catch::TestRunOrder::LexicographicallySorted;
    session.useConfigData(config);

    return session.run(argc, argv);
}
